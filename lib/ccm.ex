defmodule CCM do
  @moduledoc """
  Convergent Cross Mapping (CCM) implementation for detecting causality in coupled nonlinear
  dynamical systems.

  CCM is a nonlinear time-series method introduced by Sugihara et al. (2012) that detects
  causal relationships between variables without requiring a parametric model.  The key
  insight is that if variable X causally influences variable Y, then the historical values
  of Y contain enough information to recover the state of X — and this recovery improves
  as more data are added (i.e. the cross-map skill *converges* with library size).

  ## Typical workflow

      # 1. Build a CCM struct for the two time series
      ccm = CCM.new(x_series, y_series, embedding_dim: 3, tau: 1)

      # 2. (Optional) select the best embedding dimension automatically
      %{optimal_dim: e} = CCM.optimal_embedding_dim(x_series)
      ccm = CCM.new(x_series, y_series, embedding_dim: e)

      # 3. Test causality in both directions simultaneously
      %{x_causes_y: fwd, y_causes_x: rev} = CCM.bidirectional_ccm(ccm)

      fwd.convergent  #=> true / false
      rev.convergent  #=> true / false
  """

  defstruct [:x_series, :y_series, :embedding_dim, :tau, :lib_sizes, :train_percent]

  @doc """
  Creates a new CCM analysis structure.

  ## Parameters
  - `x_series` – list of numeric values for variable X
  - `y_series` – list of numeric values for variable Y (must be the same length as `x_series`)
  - `opts`     – keyword list of options:
    - `:embedding_dim` – embedding dimension E used to reconstruct the attractor (default: `3`).
      Use `optimal_embedding_dim/2` to choose this automatically.
    - `:tau`           – time delay between successive embedding coordinates (default: `1`).
      Values > 1 are useful when the time series is oversampled.
    - `:train_percent` – fraction of the embedded series used as the training library
      (default: `0.75`, matching `skccm`).
    - `:lib_sizes`     – explicit list of library sizes to evaluate.  When omitted, sizes are
      generated automatically up to the training-library length.

  ## Returns
  A `%CCM{}` struct ready to be passed to `cross_map/2` or `bidirectional_ccm/1`.

  ## Raises
  - `ArgumentError` if `x_series` and `y_series` have different lengths.
  - `ArgumentError` if `:embedding_dim` or `:tau` is less than 1.
  - `ArgumentError` if `:train_percent` is not between 0 and 1.

  ## Examples

      iex> CCM.new([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) |> Map.get(:embedding_dim)
      3

      iex> CCM.new([1, 2, 3], [1, 2], []) |> Map.get(:embedding_dim)
      ** (ArgumentError) x_series and y_series must have the same length
  """
  def new(x_series, y_series, opts \\ [])

  def new(x_series, y_series, _opts) when length(x_series) != length(y_series) do
    raise ArgumentError, "x_series and y_series must have the same length"
  end

  def new(x_series, y_series, opts) do
    embedding_dim = Keyword.get(opts, :embedding_dim, 3)
    tau = Keyword.get(opts, :tau, 1)
    train_percent = Keyword.get(opts, :train_percent, 0.75)

    if Keyword.has_key?(opts, :num_samples) do
      raise ArgumentError, "num_samples is unsupported; CCM uses a chronological train/test split"
    end

    if embedding_dim < 1 or tau < 1 do
      raise ArgumentError, "embedding_dim and tau must be >= 1"
    end

    if not is_number(train_percent) or train_percent <= 0 or train_percent >= 1 do
      raise ArgumentError, "train_percent must be greater than 0 and less than 1"
    end

    embedding_length = length(x_series) - (embedding_dim - 1) * tau
    train_length = trunc(embedding_length * train_percent)
    lib_sizes = Keyword.get(opts, :lib_sizes) || generate_lib_sizes(train_length)

    %CCM{
      x_series: x_series,
      y_series: y_series,
      embedding_dim: embedding_dim,
      tau: tau,
      lib_sizes: lib_sizes,
      train_percent: train_percent
    }
  end

  @doc """
  Performs CCM analysis for a single causal direction.

  Both time series are embedded and split chronologically. The first `train_percent`
  of each embedded series forms the training library; remaining points are held out
  for testing. For each library size, the first points of the training set are used.
  Neighbours are found on the source manifold and their weights are applied to the
  corresponding embedded vectors of the target series. The skill is the mean Pearson
  correlation across embedding coordinates, matching `skccm`.

  ## Parameters
  - `ccm`       – a `%CCM{}` struct produced by `new/3`
  - `direction` – atom indicating the causal direction to test:
    - `:x_causes_y` (default) – tests whether X drives Y by cross-mapping Y's manifold
      to recover X
    - `:y_causes_x` – tests whether Y drives X by cross-mapping X's manifold to recover Y

  ## Returns
  A map with:
  - `:direction`  – the direction atom passed in
  - `:results`    – list of `{library_size, mean_correlation}` tuples, one per lib size
  - `:convergent` – `true` if the skill shows a significant positive trend and the peak
    skill exceeds 0.3 (Sugihara et al. 2012 criterion), `false` otherwise
  """
  def cross_map(%CCM{} = ccm, direction \\ :x_causes_y) do
    {source_series, target_series} =
      case direction do
        :x_causes_y -> {ccm.y_series, ccm.x_series}
        :y_causes_x -> {ccm.x_series, ccm.y_series}
      end

    source_embedding = time_delay_embedding(source_series, ccm.embedding_dim, ccm.tau)
    target_embedding = time_delay_embedding(target_series, ccm.embedding_dim, ccm.tau)
    train_length = trunc(length(source_embedding) * ccm.train_percent)
    {source_train, source_test} = Enum.split(source_embedding, train_length)
    {target_train, target_test} = Enum.split(target_embedding, train_length)

    results =
      Enum.map(ccm.lib_sizes, fn lib_size ->
        skill =
          cross_map_sample(
            source_train,
            target_train,
            source_test,
            target_test,
            lib_size
          )

        {lib_size, skill}
      end)

    %{
      direction: direction,
      results: results,
      convergent: convergent?(results)
    }
  end

  @doc """
  Determines the optimal embedding dimension for a time series using simplex projection
  with leave-one-out (LOO) cross-validation (Sugihara & May, 1990).

  For each candidate embedding dimension from 1 to `max_dim`, the series is embedded
  with the given time delay `tau`, and each point is predicted from E+1 nearest neighbours
  drawn from the rest of the library (LOO).  The Pearson correlation between predictions
  and actuals is used as the skill score.  The embedding dimension that maximises this
  skill is returned as the optimal choice.

  ## Parameters
  - `series`  – list of numeric values
  - `opts`    – keyword list of options:
    - `:tau`     – time delay (default: 1)
    - `:max_dim` – largest embedding dimension to test (default: 10)

  ## Returns
  A map with:
  - `:optimal_dim` – the embedding dimension with the highest LOO skill
  - `:skills`      – list of `{embedding_dim, skill}` tuples for every dimension tested

  ## Examples

      iex> {x, _y} = CoupledLogisticMapsGenerator.run(200, 0.3)
      iex> %{optimal_dim: e} = CCM.optimal_embedding_dim(x)
      iex> e in 1..10
      true
  """
  def optimal_embedding_dim(series, opts \\ []) do
    tau = Keyword.get(opts, :tau, 1)
    max_dim = Keyword.get(opts, :max_dim, 10)

    if tau < 1, do: raise(ArgumentError, "tau must be >= 1")
    if max_dim < 1, do: raise(ArgumentError, "max_dim must be >= 1")

    skills =
      Enum.map(1..max_dim, fn e ->
        {e, simplex_loo_skill(series, e, tau)}
      end)

    {optimal_dim, _best_skill} = Enum.max_by(skills, &elem(&1, 1))

    %{optimal_dim: optimal_dim, skills: skills}
  end

  @doc """
  Performs CCM analysis in both causal directions concurrently.

  Runs `:x_causes_y` and `:y_causes_x` as parallel `Task`s and collects the results.
  This is the recommended entry point for a full causality analysis because genuine
  unidirectional causality produces convergence in only one direction, while bidirectional
  coupling produces convergence in both.

  ## Parameters
  - `ccm` – a `%CCM{}` struct produced by `new/3`

  ## Returns
  A map with two keys, each containing the result map from `cross_map/2`:
  - `:x_causes_y` – result for the X → Y direction
  - `:y_causes_x` – result for the Y → X direction

  ## Examples

      iex> {x, y} = CoupledLogisticMapsGenerator.run(300, 0.15)
      iex> ccm = CCM.new(x, y, embedding_dim: 3, tau: 1)
      iex> %{x_causes_y: fwd, y_causes_x: rev} = CCM.bidirectional_ccm(ccm)
      iex> is_boolean(fwd.convergent) and is_boolean(rev.convergent)
      true
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

  defp simplex_loo_skill(series, embedding_dim, tau) do
    embedding = time_delay_embedding(series, embedding_dim, tau)
    adjusted = Enum.drop(series, (embedding_dim - 1) * tau)
    n = length(embedding)

    # Need at least E+2 points so that LOO leaves E+1 neighbours available
    if n < embedding_dim + 2 do
      0.0
    else
      embedding_arr = List.to_tuple(embedding)
      target_arr = List.to_tuple(adjusted)

      predictions =
        for i <- 0..(n - 1) do
          library = for j <- 0..(n - 1), j != i, do: elem(embedding_arr, j)
          lib_targets = for j <- 0..(n - 1), j != i, do: elem(target_arr, j)
          predicted = predict_point(elem(embedding_arr, i), library, lib_targets)
          {elem(target_arr, i), predicted}
        end

      correlation(predictions)
    end
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

  defp cross_map_sample([], _target_train, _source_test, _target_test, _lib_size), do: 0.0

  defp cross_map_sample(source_train, target_train, source_test, target_test, lib_size) do
    library_size = min(lib_size, length(source_train))
    neighbor_count = length(hd(source_train)) + 1

    if library_size < neighbor_count or length(source_test) < 2 do
      0.0
    else
      source_library = Enum.take(source_train, library_size)
      target_library = Enum.take(target_train, library_size)

      coordinate_predictions =
        Enum.zip(source_test, target_test)
        |> Enum.map(fn {source_point, target_point} ->
          {target_point, predict_manifold_point(source_point, source_library, target_library)}
        end)
        |> Enum.reduce(List.duplicate([], length(hd(target_train))), fn {actual, predicted},
                                                                        acc ->
          Enum.zip(acc, Enum.zip(actual, predicted))
          |> Enum.map(fn {values, pair} -> [pair | values] end)
        end)

      coordinate_predictions
      |> Enum.map(fn reversed_pairs -> correlation(Enum.reverse(reversed_pairs)) end)
      |> mean()
    end
  end

  defp predict_manifold_point(query_point, source_library, target_library) do
    nearest =
      source_library
      |> Enum.with_index()
      |> Enum.map(fn {point, idx} -> {euclidean_distance(query_point, point), idx} end)
      |> Enum.sort_by(&elem(&1, 0))
      |> Enum.take(length(query_point) + 1)

    weights = calculate_weights(nearest)

    Enum.with_index(hd(target_library))
    |> Enum.map(fn {_value, coordinate} ->
      nearest
      |> Enum.zip(weights)
      |> Enum.reduce(0.0, fn {{_distance, idx}, weight}, sum ->
        sum + Enum.at(Enum.at(target_library, idx), coordinate) * weight
      end)
    end)
  end

  defp predict_point(_query_point, [], _lib_targets), do: 0.0
  defp predict_point([], _library, _lib_targets), do: 0.0

  defp predict_point(query_point, library, lib_targets) do
    distances =
      Enum.map(Enum.with_index(library), fn {lib_point, idx} ->
        dist = euclidean_distance(query_point, lib_point)
        {dist, idx}
      end)

    nearest =
      distances
      |> Enum.sort_by(&elem(&1, 0))
      |> Enum.take(min(length(query_point) + 1, length(library)))

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

  defp mean([]), do: 0.0
  defp mean(values), do: Enum.sum(values) / length(values)

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

    weights = Enum.map(dist_values, fn d -> :math.exp(-d / (min_dist + 1.0e-5)) end)
    sum = Enum.sum(weights)

    Enum.map(weights, fn weight -> weight / sum end)
  end

  @doc false
  # Expose a tiny wrapper for testing predict_point behavior without making the internal
  # implementation public in the main API. Intended for test coverage only.
  def predict_point_for_tests(query_point, library, lib_targets) do
    predict_point(query_point, library, lib_targets)
  end
end
