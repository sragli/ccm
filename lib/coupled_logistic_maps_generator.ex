defmodule CoupledLogisticMapsGenerator do
  @moduledoc """
  Generates coupled logistic maps for testing.
  """

  def run(length, coupling_strength \\ 0.02) do
    r1 = 3.7
    r2 = 3.6

    {x_list, y_list} =
      {0.1, 0.2}
      |> Stream.iterate(fn {x_prev, y_prev} ->
        # Coupled logistic maps with proper bounds checking
        x_raw = r1 * x_prev * (1 - x_prev) + coupling_strength * (y_prev - x_prev)
        y_raw = r2 * y_prev * (1 - y_prev)

        # Clamp values to [0, 1] to maintain stability
        {max(0.0, min(1.0, x_raw)), max(0.0, min(1.0, y_raw))}
      end)
      |> Stream.take(length)
      |> Enum.to_list()
      |> Enum.unzip()

    {x_list, y_list}
  end
end
