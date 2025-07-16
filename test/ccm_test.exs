defmodule CCMTest do
  use ExUnit.Case
  doctest CCM

  test "computes correct results" do
    {x_series, y_series} = CCM.generate_coupled_logistic_maps(300, 0.15)

    ccm = CCM.new(x_series, y_series, embedding_dim: 3, tau: 1, num_samples: 50)

    assert %{x_causes_y: %{direction: :x_causes_y, convergent: false},
             y_causes_x: %{direction: :y_causes_x, convergent: false}} = CCM.bidirectional_ccm(ccm)
  end
end
