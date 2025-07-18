defmodule CCM do
  @moduledoc """
  Convergent Cross Mapping (CCM) implementation for detecting causality
  in coupled nonlinear dynamical systems.
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

    # Create embedding from source series
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
      convergent: is_convergent?(results)
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

  defp generate_lib_sizes(max_size) do
    if max_size < 10 do
      [max_size]
    else
      start_size = max(5, div(max_size, 10))
      step = max(2, div(max_size, 20))

      Stream.iterate(start_size, &(&1 + step))
      |> Stream.take_while(&(&1 <= max_size))
      |> Enum.to_list()
    end
  end

  defp time_delay_embedding(series, embedding_dim, tau) do
    max_index = length(series) - (embedding_dim - 1) * tau

    for i <- 0..(max_index - 1) do
      for j <- 0..(embedding_dim - 1) do
        Enum.at(series, i + j * tau)
      end
    end
  end

  defp cross_map_sample(embedding, target_series, lib_size, embedding_dim, tau) do
    # Randomly sample library points
    total_points = length(embedding)

    # Ensure lib_size doesn't exceed available points
    actual_lib_size = min(lib_size, total_points - 1)
    if actual_lib_size < 1, do: 0.0

    lib_indices = Enum.take_random(0..(total_points - 1), actual_lib_size)
    library = Enum.map(lib_indices, &Enum.at(embedding, &1))

    # Adjust target series for embedding offset
    adjusted_target = Enum.drop(target_series, (embedding_dim - 1) * tau)

    # Ensure we have enough target values
    if length(adjusted_target) < total_points, do: 0.0

    lib_targets = Enum.map(lib_indices, &Enum.at(adjusted_target, &1))

    # Predict each point not in library
    pred_indices = Enum.to_list(0..(total_points - 1)) -- lib_indices

    if length(pred_indices) < 2, do: 0.0

    predictions =
      Enum.map(pred_indices, fn pred_idx ->
        query_point = Enum.at(embedding, pred_idx)
        actual_value = Enum.at(adjusted_target, pred_idx)
        predicted_value = predict_point(query_point, library, lib_targets)
        {actual_value, predicted_value}
      end)

    # Filter out nil values
    valid_predictions =
      predictions
      |> Enum.filter(fn {actual, predicted} ->
        actual != nil and predicted != nil and
          is_number(actual) and is_number(predicted)
      end)

    if length(valid_predictions) < 2 do
      0.0
    else
      # Calculate correlation
      correlation(valid_predictions)
    end
  end

  defp predict_point(query_point, library, lib_targets) do
    if length(library) == 0 or length(lib_targets) == 0, do: 0.0

    # Find k+1 nearest neighbors (where k is embedding dimension)
    k = min(length(query_point), length(library) - 1)
    if k < 1, do: 0.0

    distances =
      Enum.map(Enum.with_index(library), fn {lib_point, idx} ->
        dist = euclidean_distance(query_point, lib_point)
        {dist, idx}
      end)

    nearest =
      distances
      |> Enum.sort_by(&elem(&1, 0))
      |> Enum.take(k + 1)

    if length(nearest) == 0, do: 0.0

    # Calculate weights using exponential kernel
    weights =
      Enum.map(nearest, fn {dist, _} ->
        if dist == 0, do: 1.0, else: :math.exp(-dist / (dist + 1.0e-8))
      end)

    total_weight = Enum.sum(weights)

    if total_weight == 0, do: 0.0

    # Weighted prediction
    weighted_sum =
      nearest
      |> Enum.zip(weights)
      |> Enum.map(fn {{_, idx}, weight} ->
        target_val = Enum.at(lib_targets, idx)
        if is_number(target_val), do: target_val * weight, else: 0.0
      end)
      |> Enum.sum()

    prediction = weighted_sum / total_weight

    # Return 0 if prediction is invalid
    if is_finite(prediction), do: prediction, else: 0.0
  end

  defp euclidean_distance(point1, point2) do
    point1
    |> Enum.zip(point2)
    |> Enum.map(fn {x, y} -> (x - y) * (x - y) end)
    |> Enum.sum()
    |> :math.sqrt()
  end

  defp correlation(predictions) do
    if length(predictions) < 2, do: 0.0

    {actuals, predicted} = Enum.unzip(predictions)

    # Filter out any invalid values (NaN shows up as :nan, infinity as :inf/-:inf)
    valid_pairs =
      actuals
      |> Enum.zip(predicted)
      |> Enum.filter(fn {a, p} ->
        is_number(a) and is_number(p) and
          a != :nan and p != :nan and
          a != :inf and p != :inf and
          a != :"-inf" and p != :"-inf" and
          is_finite(a) and is_finite(p)
      end)

    if length(valid_pairs) < 2, do: 0.0

    {valid_actuals, valid_predicted} = Enum.unzip(valid_pairs)

    actual_mean = Enum.sum(valid_actuals) / length(valid_actuals)
    pred_mean = Enum.sum(valid_predicted) / length(valid_predicted)

    numerator =
      valid_actuals
      |> Enum.zip(valid_predicted)
      |> Enum.map(fn {a, p} -> (a - actual_mean) * (p - pred_mean) end)
      |> Enum.sum()

    actual_var =
      valid_actuals
      |> Enum.map(fn a -> (a - actual_mean) * (a - actual_mean) end)
      |> Enum.sum()

    pred_var =
      valid_predicted
      |> Enum.map(fn p -> (p - pred_mean) * (p - pred_mean) end)
      |> Enum.sum()

    denominator = :math.sqrt(actual_var * pred_var)

    cond do
      denominator == 0 ->
        0.0

      not is_finite(denominator) ->
        0.0

      true ->
        result = numerator / denominator
        if is_finite(result), do: result, else: 0.0
    end
  end

  # Helper function to check if a number is finite
  defp is_finite(x) when is_number(x) do
    x != :inf and x != :"-inf" and x != :nan and
      x > -1.0e308 and x < 1.0e308
  end

  defp is_finite(_), do: false

  defp is_convergent?(results) do
    if length(results) < 3, do: false

    # Filter out invalid correlations
    valid_results =
      results
      |> Enum.filter(fn {_, corr} ->
        is_number(corr) and is_finite(corr)
      end)

    if length(valid_results) < 3, do: false

    {valid_lib_sizes, valid_correlations} = Enum.unzip(valid_results)

    n = length(valid_results)
    sum_x = Enum.sum(valid_lib_sizes)
    sum_y = Enum.sum(valid_correlations)

    sum_xy =
      valid_lib_sizes
      |> Enum.zip(valid_correlations)
      |> Enum.map(fn {x, y} -> x * y end)
      |> Enum.sum()

    sum_x2 = valid_lib_sizes |> Enum.map(fn x -> x * x end) |> Enum.sum()

    denominator = n * sum_x2 - sum_x * sum_x

    if denominator == 0 do
      false
    else
      slope = (n * sum_xy - sum_x * sum_y) / denominator
      # Positive slope indicates convergence
      slope > 0.001
    end
  end
end
