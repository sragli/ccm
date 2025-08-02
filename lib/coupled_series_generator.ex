defmodule CoupledSeriesGenerator do
  @moduledoc """
  Generates coupled time series where X forces Y for testing CCM implementation.
  """

  @doc """
  Generates a driving time series X and driven time series Y.
  X follows a chaotic logistic map, Y is forced by X with some coupling strength.
  """
  def generate_coupled_series(length \\ 50, opts \\ []) do
    # Parameters
    # Chaos parameter for X
    r_x = Keyword.get(opts, :r_x, 3.8)
    # Chaos parameter for Y
    r_y = Keyword.get(opts, :r_y, 3.6)
    # Coupling strength (X -> Y)
    coupling = Keyword.get(opts, :coupling, 0.3)
    # Observation noise
    noise_level = Keyword.get(opts, :noise_level, 0.05)

    # Initial conditions
    x0 = Keyword.get(opts, :x0, 0.3)
    y0 = Keyword.get(opts, :y0, 0.4)

    # Generate the coupled system
    {x_series, y_series} = generate_system(length, r_x, r_y, coupling, x0, y0)

    # Add observation noise
    x_noisy = add_noise(x_series, noise_level)
    y_noisy = add_noise(y_series, noise_level)

    %{
      x_series: x_noisy,
      y_series: y_noisy,
      parameters: %{
        r_x: r_x,
        r_y: r_y,
        coupling: coupling,
        noise_level: noise_level,
        length: length
      },
      description: "X drives Y with coupling strength #{coupling}"
    }
  end

  defp generate_system(length, r_x, r_y, coupling, x0, y0) do
    # Use Stream.scan to generate the coupled dynamics
    initial_state = {x0, y0}

    states =
      1..length
      |> Stream.scan(initial_state, fn _, {x_prev, y_prev} ->
        # X evolves according to logistic map (autonomous)
        x_next = r_x * x_prev * (1 - x_prev)

        # Y is forced by X (coupled system)
        # Y equation: y_next = r_y * y_prev * (1 - y_prev) + coupling * (x_prev - y_prev)
        y_autonomous = r_y * y_prev * (1 - y_prev)
        y_forcing = coupling * (x_prev - y_prev)
        y_next = y_autonomous + y_forcing

        # Keep values in reasonable bounds
        x_next = max(0.001, min(0.999, x_next))
        y_next = max(0.001, min(0.999, y_next))

        {x_next, y_next}
      end)
      |> Enum.to_list()

    # Add initial condition and extract series
    all_states = [initial_state | states]
    x_series = Enum.map(all_states, fn {x, _} -> x end)
    y_series = Enum.map(all_states, fn {_, y} -> y end)

    {x_series, y_series}
  end

  defp add_noise(series, noise_level) do
    Enum.map(series, fn value ->
      noise = noise_level * (:rand.uniform() - 0.5) * 2
      value + noise
    end)
  end

  @doc """
  Generates test data with known causality for CCM validation.
  """
  def generate_test_cases do
    [
      # Strong coupling: X clearly drives Y
      generate_coupled_series(50, coupling: 0.4, noise_level: 0.02),

      # Medium coupling: Moderate causality
      generate_coupled_series(50, coupling: 0.2, noise_level: 0.05),

      # Weak coupling: Subtle causality
      generate_coupled_series(50, coupling: 0.1, noise_level: 0.03),

      # No coupling: Independent systems (negative control)
      generate_coupled_series(50, coupling: 0.0, noise_level: 0.05)
    ]
  end

  @doc """
  Pretty prints the generated series for inspection.
  """
  def print_series(%{x_series: x_series, y_series: y_series, description: desc}) do
    IO.puts("=== #{desc} ===")
    IO.puts("X series (first 10): #{inspect(Enum.take(x_series, 10))}")
    IO.puts("Y series (first 10): #{inspect(Enum.take(y_series, 10))}")
    IO.puts("X length: #{length(x_series)}, Y length: #{length(y_series)}")
    IO.puts("")
  end

  def run(coupling \\ 0.03) do
    # Set seed for reproducibility
    :rand.seed(:exsss, {1, 2, 3})

    # Generate a single coupled system
    example_data =
      CoupledSeriesGenerator.generate_coupled_series(50,
        coupling: coupling,
        noise_level: 0.03
      )

    CoupledSeriesGenerator.print_series(example_data)

    # The actual 50-element series you requested:
    IO.puts("=== 50-ELEMENT COUPLED TIME SERIES ===")
    IO.puts("X series (driving): #{inspect(example_data.x_series, limit: :infinity)}")
    IO.puts("")
    IO.puts("Y series (driven): #{inspect(example_data.y_series, limit: :infinity)}")

    # Quick verification
    IO.puts("\n=== VERIFICATION ===")

    IO.puts(
      "Both series have #{length(example_data.x_series)} elements: #{length(example_data.x_series) == 50}"
    )

    IO.puts("Coupling strength: #{example_data.parameters.coupling}")
    IO.puts("Expected CCM result: X causes Y should show convergence, Y causes X should not")
  end
end
