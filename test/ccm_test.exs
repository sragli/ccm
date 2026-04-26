defmodule CCMTest do
  use ExUnit.Case
  doctest CCM

  test "computes correct results" do
    {x_series, y_series} = CoupledLogisticMapsGenerator.run(300, 0.15)

    ccm = CCM.new(x_series, y_series, embedding_dim: 3, tau: 1, num_samples: 50)

    assert %{
             x_causes_y: %{direction: :x_causes_y, convergent: false},
             y_causes_x: %{direction: :y_causes_x, convergent: true}
           } = CCM.bidirectional_ccm(ccm)
  end

  test "new/3 raises when series lengths differ" do
    assert_raise ArgumentError, fn ->
      CCM.new([1, 2, 3], [1, 2], [])
    end
  end

  test "new uses single lib_size when max_size < 10 and defaults are set" do
    # length = 11 -> embedding_length = 11 - (3-1)*1 = 9, auto_max = 9 - (3+2) = 4 (< 10)
    series = Enum.to_list(1..11)
    ccm = CCM.new(series, series)

    assert ccm.lib_sizes == [4]
    assert ccm.embedding_dim == 3
    assert ccm.tau == 1
    assert ccm.num_samples == 100
  end

  test "cross_map returns expected shape and types" do
    x = Enum.to_list(1..50)
    # y is a linear function of x (deterministic relationship)
    y = Enum.map(x, &(&1 * 2))

    ccm = CCM.new(x, y, num_samples: 5, lib_sizes: [5, 10, 15])

    result = CCM.cross_map(ccm)

    assert result.direction == :x_causes_y
    assert is_list(result.results)
    assert is_boolean(result.convergent)

    for {lib_size, avg_corr} <- result.results do
      assert is_integer(lib_size)
      assert is_float(avg_corr)
    end
  end

  test "cross_map returns zero correlation when lib_size >= embedding length" do
    # small series where embedding length is 4
    series = Enum.to_list(1..6)
    # choose a lib_size bigger than embedding length
    ccm = CCM.new(series, series, lib_sizes: [10], num_samples: 3)

    %{results: results} = CCM.cross_map(ccm)

    assert results == [{10, 0.0}]
  end

  test "bidirectional_ccm returns both direction maps" do
    x = Enum.to_list(1..30)
    y = Enum.map(x, &(&1 + 1))

    ccm = CCM.new(x, y, num_samples: 3)

    out = CCM.bidirectional_ccm(ccm)

    assert Map.has_key?(out, :x_causes_y)
    assert Map.has_key?(out, :y_causes_x)

    assert out.x_causes_y.direction == :x_causes_y
    assert out.y_causes_x.direction == :y_causes_x
  end

  test "handles very short series gracefully" do
    # series too short to build embeddings with embedding_dim 3 and tau 1
    short = [1, 2, 3]

    ccm = CCM.new(short, short, embedding_dim: 3, tau: 1, lib_sizes: [1], num_samples: 2)

    # cross_map should run but produce results (likely zeros) without crashing
    res = CCM.cross_map(ccm)
    assert is_list(res.results)
  end

  test "lib_size zero and empty library predictions" do
    series = Enum.to_list(1..20)
    # lib_size 0 should lead to no library points -> predict_point should return 0
    ccm = CCM.new(series, series, lib_sizes: [0], num_samples: 3)

    %{results: results} = CCM.cross_map(ccm)
    # correlation should be 0.0 when predictions are not possible
    assert results == [{0, 0.0}]
  end

  test "cross_map handles identical points without crashing" do
    # Create a series with repeated identical values so distances can be zero
    x = Enum.map(1..30, fn i -> if rem(i, 3) == 0, do: 1.0, else: i * 1.0 end)
    y = Enum.map(x, &(&1 + 0.5))

    ccm = CCM.new(x, y, num_samples: 5, lib_sizes: [5, 8])

    result = CCM.cross_map(ccm)

    assert is_list(result.results)

    for {_lib_size, avg_corr} <- result.results do
      assert is_float(avg_corr)
    end
  end

  test "predict_point_for_tests returns exact target for identical neighbor" do
    query = [1.0, 2.0, 3.0]
    library = [[1.0, 2.0, 3.0], [10.0, 10.0, 10.0]]
    targets = [5.0, 100.0]

    pred = CCM.predict_point_for_tests(query, library, targets)

    # Since the first library point equals the query, it should dominate the prediction
    assert_in_delta(pred, 5.0, 1.0e-6)
  end

  test "predict_point_for_tests uses exponential weights" do
    # query equidistant between two library points -> prediction should be average of targets
    query = [0.0, 0.0]
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    library = [a, b]
    targets = [10.0, 20.0]

    pred = CCM.predict_point_for_tests(query, library, targets)

    assert_in_delta(pred, 15.0, 1.0e-6)
  end

  test "optimal_embedding_dim returns a map with expected keys" do
    {x, _y} = CoupledLogisticMapsGenerator.run(200, 0.3)

    result = CCM.optimal_embedding_dim(x)

    assert Map.has_key?(result, :optimal_dim)
    assert Map.has_key?(result, :skills)
    assert is_integer(result.optimal_dim)
    assert result.optimal_dim in 1..10
    assert length(result.skills) == 10

    for {e, skill} <- result.skills do
      assert is_integer(e)
      assert is_float(skill)
    end
  end

  test "optimal_embedding_dim respects max_dim option" do
    {x, _y} = CoupledLogisticMapsGenerator.run(100, 0.2)

    result = CCM.optimal_embedding_dim(x, max_dim: 5)

    assert length(result.skills) == 5
    assert result.optimal_dim in 1..5
  end

  test "optimal_embedding_dim respects tau option" do
    {x, _y} = CoupledLogisticMapsGenerator.run(150, 0.2)

    result = CCM.optimal_embedding_dim(x, tau: 2, max_dim: 5)

    assert length(result.skills) == 5
    assert result.optimal_dim in 1..5
  end

  test "optimal_embedding_dim raises for invalid tau" do
    assert_raise ArgumentError, fn ->
      CCM.optimal_embedding_dim([1.0, 2.0, 3.0], tau: 0)
    end
  end

  test "optimal_embedding_dim raises for invalid max_dim" do
    assert_raise ArgumentError, fn ->
      CCM.optimal_embedding_dim([1.0, 2.0, 3.0], max_dim: 0)
    end
  end

  test "optimal_embedding_dim handles series too short for high E gracefully" do
    # 8 points; high E will produce 0.0 skill but should not crash
    short = Enum.map(1..8, &(&1 * 1.0))

    result = CCM.optimal_embedding_dim(short, max_dim: 6)

    assert is_integer(result.optimal_dim)
    for {_e, skill} <- result.skills, do: assert(is_float(skill))
  end
end
