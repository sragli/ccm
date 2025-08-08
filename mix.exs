defmodule CCM.MixProject do
  use Mix.Project

  def project do
    [
      app: :ccm,
      version: "0.1.0",
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      description: description(),
      package: package(),
      deps: deps(),
      name: "CCM",
      source_url: "https://github.com/sragli/ccm",
      docs: docs()
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp description() do
    "Elixir module that implements Convergent Cross Mapping (CCM) for time series data."
  end

  defp package() do
    [
      files: ~w(lib .formatter.exs mix.exs README.md LICENSE CHANGELOG),
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => "https://github.com/sragli/ccm"}
    ]
  end

  defp docs() do
    [
      main: "CCM",
      extras: ["README.md", "LICENSE", "examples.livemd"]
    ]
  end

  defp deps do
    [
      {:ex_doc, ">= 0.0.0", only: :dev, runtime: false}
    ]
  end
end
