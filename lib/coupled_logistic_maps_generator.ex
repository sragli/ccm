defmodule CoupledLogisticMapsGenerator do
  @moduledoc """
  Generates coupled logistic maps for testing.
  """

  def run(length, coupling_strength \\ 0.02) do
    r1 = 3.7
    r2 = 3.6

    {x_series, y_series} =
      Enum.reduce(1..length, {[0.1], [0.2]}, fn _, {x_acc, y_acc} ->
        x_prev = hd(x_acc)
        y_prev = hd(y_acc)

        # Coupled logistic maps with proper bounds checking
        x_raw = r1 * x_prev * (1 - x_prev) + coupling_strength * (y_prev - x_prev)
        y_raw = r2 * y_prev * (1 - y_prev)

        # Clamp values to [0, 1] to maintain stability
        x_new = max(0.0, min(1.0, x_raw))
        y_new = max(0.0, min(1.0, y_raw))

        {[x_new | x_acc], [y_new | y_acc]}
      end)

    {Enum.reverse(x_series), Enum.reverse(y_series)}
  end
end
