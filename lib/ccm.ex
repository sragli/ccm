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
  def new(x_series, y_series, opts \\ [])

  def new(x_series, y_series, _opts) when length(x_series) != length(y_series) do
    raise ArgumentError, "x_series and y_series must have the same length"
  end

  def new(x_series, y_series, opts) do
    embedding_dim = Keyword.get(opts, :embedding_dim, 3)
    tau = Keyword.get(opts, :tau, 1)
    num_samples = Keyword.get(opts, :num_samples, 100)

    # basic validation and clamping
    if embedding_dim < 1 or tau < 1 do
      raise ArgumentError, "embedding_dim and tau must be >= 1"
    end

    if num_samples < 1 do
      raise ArgumentError, "num_samples must be >= 1"
    end

    max_lib_size = Keyword.get(opts, :lib_sizes, nil)

    # Maximum valid library size: must leave at least embedding_dim + 2 points for prediction
    # (E+1 neighbours are needed; keeping E+2 free ensures at least one independent test point)
    embedding_length = length(x_series) - (embedding_dim - 1) * tau
    auto_max = max(0, embedding_length - (embedding_dim + 2))
    lib_sizes = if max_lib_size, do: max_lib_size, else: generate_lib_sizes(auto_max)

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

    # Convert to tuples once for O(1) random access in tight inner loops.
    # adjusted_target is hoisted here so it is not recomputed on every sample.
    embedding_arr = List.to_tuple(embedding)
    target_arr = List.to_tuple(Enum.drop(target_series, (ccm.embedding_dim - 1) * ccm.tau))
    total_points = tuple_size(embedding_arr)

    results =
      Enum.map(ccm.lib_sizes, fn lib_size ->
        correlations =
          Enum.map(1..ccm.num_samples, fn _ ->
            cross_map_sample(embedding_arr, target_arr, lib_size, total_points)
          end)

        {lib_size, Enum.sum(correlations) / ccm.num_samples}
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
    # Both directions are independent, run them in parallel.
    x_task = Task.async(fn -> cross_map(ccm, :x_causes_y) end)
    y_task = Task.async(fn -> cross_map(ccm, :y_causes_x) end)

    %{
      x_causes_y: Task.await(x_task, :infinity),
      y_causes_x: Task.await(y_task, :infinity)
    }
  end

  defp generate_lib_sizes(max_size) when max_size < 10, do: [max_size]

  defp generate_lib_sizes(max_size) do
    step = max(2, div(max_size, 20))

    sizes =
      max_size
      |> div(10)
      |> max(5)
      |> Stream.iterate(&(&1 + step))
      |> Stream.take_while(&(&1 <= max_size))
      |> Enum.to_list()

    if List.last(sizes) == max_size, do: sizes, else: sizes ++ [max_size]
  end

  defp time_delay_embedding(series, embedding_dim, tau) do
    max_index = length(series) - (embedding_dim - 1) * tau

    if max_index <= 0 do
      []
    else
      for i <- 0..(max_index - 1) do
        for j <- 0..(embedding_dim - 1) do
          Enum.at(series, i + j * tau)
        end
      end
    end
  end

  defp cross_map_sample(_embedding, _target, lib_size, total_points)
       when lib_size >= total_points,
       do: 0.0

  defp cross_map_sample(embedding, target, lib_size, total_points) do
    actual_lib_size = min(lib_size, total_points - 1)

    lib_indices = Enum.take_random(0..(total_points - 1), actual_lib_size)

    if tuple_size(target) < total_points or total_points - actual_lib_size < 2 do
      0.0
    else
      # MapSet membership check is O(1) vs O(n) for list `--`
      lib_set = MapSet.new(lib_indices)

      library = Enum.map(lib_indices, &elem(embedding, &1))
      lib_targets = Enum.map(lib_indices, &elem(target, &1))

      predictions =
        for idx <- 0..(total_points - 1), not MapSet.member?(lib_set, idx) do
          {elem(target, idx), predict_point(elem(embedding, idx), library, lib_targets)}
        end

      correlation(predictions)
    end
  end

  defp predict_point(_query_point, [], _lib_targets), do: 0.0
  defp predict_point([], _library, _lib_targets), do: 0.0

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

  defp euclidean_distance(point1, point2) do
    point1
    |> Enum.zip(point2)
    |> Enum.map(fn {x, y} -> (x - y) * (x - y) end)
    |> Enum.sum()
    |> :math.sqrt()
  end

  defp correlation(predictions) when length(predictions) < 2, do: 0.0

  defp correlation(predictions) do
    n = length(predictions)
    {actuals, predicted} = Enum.unzip(predictions)

    actual_mean = Enum.sum(actuals) / n
    pred_mean = Enum.sum(predicted) / n

    # Single pass over both lists to compute covariance and both variances.
    {cov, var_a, var_p} =
      Enum.reduce(Enum.zip(actuals, predicted), {0.0, 0.0, 0.0}, fn {a, p}, {c, va, vp} ->
        da = a - actual_mean
        dp = p - pred_mean
        {c + da * dp, va + da * da, vp + dp * dp}
      end)

    denominator = :math.sqrt(var_a * var_p)
    if denominator != 0, do: cov / denominator, else: 0.0
  end

  defp convergent?(results) when length(results) < 3, do: false

  defp convergent?(results) do
    {lib_sizes, correlations} = Enum.unzip(results)

    n = length(results)
    mean_x = Enum.sum(lib_sizes) / n
    mean_y = Enum.sum(correlations) / n

    cov =
      lib_sizes
      |> Enum.zip(correlations)
      |> Enum.map(fn {x, y} -> (x - mean_x) * (y - mean_y) end)
      |> Enum.sum()

    var_x =
      lib_sizes
      |> Enum.map(fn x -> (x - mean_x) * (x - mean_x) end)
      |> Enum.sum()

    var_y =
      correlations
      |> Enum.map(fn y -> (y - mean_y) * (y - mean_y) end)
      |> Enum.sum()

    denom = :math.sqrt(var_x * var_y)

    if denom == 0 do
      false
    else
      # Pearson r > 0.5 signals a positive trend; additionally require the peak
      # cross-map skill to exceed 0.3 so that low-but-weakly-increasing noise
      # is not misclassified as convergent (Sugihara et al. 2012 criterion).
      cov / denom > 0.5 and Enum.max(correlations) > 0.3
    end
  end

  defp calculate_weights([]), do: []

  defp calculate_weights(distances) do
    # distances is a list of {dist, idx}
    dist_values = Enum.map(distances, fn {dist, _} -> dist end)
    min_dist = Enum.min(dist_values)

    # Sugihara et al. (2012) exponential weighting: w_i = exp(-d_i / d_min)
    # If any distance is effectively zero, give full weight to exact matches only
    if min_dist < 1.0e-12 do
      Enum.map(dist_values, fn d -> if d < 1.0e-12, do: 1.0, else: 0.0 end)
    else
      weights = Enum.map(dist_values, fn d -> :math.exp(-d / min_dist) end)
      sum = Enum.sum(weights)

      if sum == 0.0 do
        Enum.map(weights, fn _ -> 0.0 end)
      else
        Enum.map(weights, fn w -> w / sum end)
      end
    end
  end

  @doc false
  # Expose a tiny wrapper for testing predict_point behavior without making the internal
  # implementation public in the main API. Intended for test coverage only.
  def predict_point_for_tests(query_point, library, lib_targets) do
    predict_point(query_point, library, lib_targets)
  end
end
