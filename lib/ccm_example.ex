defmodule CCMExample do
  import CCM

  @doc """
  Utility function to generate coupled logistic maps for testing.
  """
  def generate_coupled_logistic_maps(length, coupling_strength \\ 0.02) do
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

  @doc """
  Example usage and testing function.
  """
  def run do
    # Generate test data with known causal relationship
    {x_series, y_series} = generate_coupled_logistic_maps(300, 0.05)

    # Create CCM analysis
    ccm = CCM.new(x_series, y_series, embedding_dim: 3, tau: 1, num_samples: 30)

    # Perform bidirectional analysis
    results = bidirectional_ccm(ccm)

    IO.puts("=== CCM Analysis Results ===")
    IO.puts("Y causes X (should be weak):")

    Enum.each(results.x_causes_y.results, fn {lib_size, corr} ->
      IO.puts("  Library size #{lib_size}: correlation = #{Float.round(corr, 4)}")
    end)

    IO.puts("  Convergent: #{results.x_causes_y.convergent}")

    IO.puts("\nX causes Y (should be strong and convergent):")

    Enum.each(results.y_causes_x.results, fn {lib_size, corr} ->
      IO.puts("  Library size #{lib_size}: correlation = #{Float.round(corr, 4)}")
    end)

    IO.puts("  Convergent: #{results.y_causes_x.convergent}")

    results
  end
end
