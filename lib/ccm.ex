defmodule CCM do
  @moduledoc """
  Convergent Cross Mapping (CCM) implementation for detecting causality in coupled nonlinear
  dynamical systems.
  """

  defstruct [:x_series, :y_series, :embedding_dim, :tau, :lib_sizes, :num_samples]

  @doc """
  Creates a new CCM analysis structure.

  ## Parameters
  - x_series: List of numeric values for variable X
  - y_series: List of numeric values for variable Y
  - embedding_dim: Embedding dimension (default: 3)
  - tau: Time delay (default: 1)
  - lib_sizes: List of library sizes to test (default: auto-generated)
  - num_samples: Number of bootstrap samples (default: 100)
  """
  def new(x_series, y_series, opts \\ []) do
    embedding_dim = Keyword.get(opts, :embedding_dim, 3)
    tau = Keyword.get(opts, :tau, 1)
    num_samples = Keyword.get(opts, :num_samples, 100)

    if length(x_series) != length(y_series) do
      raise ArgumentError, "x_series and y_series must have the same length"
    end

    max_lib_size = length(x_series) - (embedding_dim - 1) * tau
    lib_sizes = Keyword.get(opts, :lib_sizes, generate_lib_sizes(max_lib_size))

    %CCM{
      x_series: x_series,
      y_series: y_series,
      embedding_dim: embedding_dim,
      tau: tau,
      lib_sizes: lib_sizes,
      num_samples: num_samples
    }
  end

  @doc """
  Performs CCM analysis to test if X causes Y.
  Returns a map with correlation coefficients for each library size.
  """
  def cross_map(%CCM{} = ccm, direction \\ :x_causes_y) do
    {source_series, target_series} =
      case direction do
        :x_causes_y -> {ccm.y_series, ccm.x_series}
        :y_causes_x -> {ccm.x_series, ccm.y_series}
      end

    embedding = time_delay_embedding(source_series, ccm.embedding_dim, ccm.tau)

    # Perform cross-mapping for each library size
    results =
      Enum.map(ccm.lib_sizes, fn lib_size ->
        correlations =
          Enum.map(1..ccm.num_samples, fn _ ->
            cross_map_sample(embedding, target_series, lib_size, ccm.embedding_dim, ccm.tau)
          end)

        avg_correlation = Enum.sum(correlations) / length(correlations)
        {lib_size, avg_correlation}
      end)

    %{
      direction: direction,
      results: results,
      convergent: convergent?(results)
    }
  end

  @doc """
  Performs bidirectional CCM analysis.
  """
  def bidirectional_ccm(%CCM{} = ccm) do
    %{
      x_causes_y: cross_map(ccm, :x_causes_y),
      y_causes_x: cross_map(ccm, :y_causes_x)
    }
  end

  defp generate_lib_sizes(max_size) when max_size < 10, do: [max_size]

  defp generate_lib_sizes(max_size) do
    step = max(2, div(max_size, 20))

    max_size
    |> div(10)
    |> max(5)
    |> Stream.iterate(&(&1 + step))
    |> Stream.take_while(&(&1 <= max_size))
    |> Enum.to_list()
  end

  defp time_delay_embedding(series, embedding_dim, tau) do
    max_index = length(series) - (embedding_dim - 1) * tau

    for i <- 0..(max_index - 1) do
      for j <- 0..(embedding_dim - 1) do
        Enum.at(series, i + j * tau)
      end
    end
  end

  defp cross_map_sample(embedding, _, lib_size, _, _) when lib_size >= length(embedding),
    do: 0.0

  defp cross_map_sample(embedding, target_series, lib_size, embedding_dim, tau) do
    total_points = length(embedding)

    actual_lib_size = min(lib_size, total_points - 1)

    lib_indices = Enum.take_random(0..(total_points - 1), actual_lib_size)
    library = Enum.map(lib_indices, &Enum.at(embedding, &1))

    adjusted_target = Enum.drop(target_series, (embedding_dim - 1) * tau)

    pred_indices = Enum.to_list(0..(total_points - 1)) -- lib_indices

    if length(adjusted_target) < total_points or length(pred_indices) < 2 do
      0.0
    else
      lib_targets = Enum.map(lib_indices, &Enum.at(adjusted_target, &1))

      predictions =
        Enum.map(pred_indices, fn pred_idx ->
          query_point = Enum.at(embedding, pred_idx)
          actual_value = Enum.at(adjusted_target, pred_idx)
          predicted_value = predict_point(query_point, library, lib_targets)
          {actual_value, predicted_value}
        end)

      correlation(predictions)
    end
  end

  defp predict_point(query_point, library, lib_targets)
       when length(library) < 1 or length(lib_targets) == 0 or length(query_point) == 0,
       do: 0.0

  defp predict_point(query_point, library, lib_targets) do
    # Use E+1 neighbors
    embedding_dim = length(query_point)
    k = min(embedding_dim + 1, length(library))

    distances =
      Enum.map(Enum.with_index(library), fn {lib_point, idx} ->
        dist = euclidean_distance(query_point, lib_point)
        {dist, idx}
      end)

    nearest =
      distances
      |> Enum.sort_by(&elem(&1, 0))
      |> Enum.take(k)

    if length(nearest) == 0 do
      0.0
    else
      weights = calculate_weights(nearest)
      total_weight = Enum.sum(weights)

      if total_weight == 0 do
        0.0
      else
        # Weighted prediction
        weighted_sum =
          nearest
          |> Enum.zip(weights)
          |> Enum.map(fn {{_, idx}, weight} -> Enum.at(lib_targets, idx) * weight end)
          |> Enum.sum()

        weighted_sum / total_weight
      end
    end
  end

  defp euclidean_distance(point1, point2) do
    point1
    |> Enum.zip(point2)
    |> Enum.map(fn {x, y} -> (x - y) * (x - y) end)
    |> Enum.sum()
    |> :math.sqrt()
  end

  defp correlation(predictions) when length(predictions) < 2, do: 0.0

  defp correlation(predictions) do
    {actuals, predicted} = Enum.unzip(predictions)

    actual_mean = Enum.sum(actuals) / length(actuals)
    pred_mean = Enum.sum(predicted) / length(predicted)

    numerator =
      actuals
      |> Enum.zip(predicted)
      |> Enum.map(fn {a, p} -> (a - actual_mean) * (p - pred_mean) end)
      |> Enum.sum()

    actual_var =
      actuals
      |> Enum.map(fn a -> (a - actual_mean) * (a - actual_mean) end)
      |> Enum.sum()

    pred_var =
      predicted
      |> Enum.map(fn p -> (p - pred_mean) * (p - pred_mean) end)
      |> Enum.sum()

    denominator = :math.sqrt(actual_var * pred_var)
    if denominator != 0, do: numerator / denominator, else: 0.0
  end

  defp convergent?(results) when length(results) < 3, do: false

  defp convergent?(results) do
    {lib_sizes, correlations} = Enum.unzip(results)

    n = length(results)
    sum_x = Enum.sum(lib_sizes)
    sum_y = Enum.sum(correlations)

    sum_xy =
      lib_sizes
      |> Enum.zip(correlations)
      |> Enum.map(fn {x, y} -> x * y end)
      |> Enum.sum()

    sum_x2 =
      lib_sizes
      |> Enum.map(fn x -> x * x end)
      |> Enum.sum()

    denominator = n * sum_x2 - sum_x * sum_x

    if denominator != 0 do
      slope = (n * sum_xy - sum_x * sum_y) / denominator
      # Positive slope indicates convergence
      slope > 0.001
    else
      false
    end
  end

  defp calculate_weights(distances) do
    # Extract just the distance values to find minimum
    dist_values = Enum.map(distances, fn {dist, _} -> dist end)
    min_dist = Enum.min(dist_values)

    dist_values
    |> Enum.map(fn dist ->
      if dist < 1.0e-12 do
        1.0
      else
        # Scale by minimum distance to avoid very small weights
        :math.exp(-dist / (min_dist + 1.0e-8))
      end
    end)
  end
end
