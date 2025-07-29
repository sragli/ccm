# CCM

Elixir module that implements Convergent Cross Mapping (CCM) for time series data.

CCM tests whether variable X causally influences variable Y by examining if historical values of Y contain information about X due to their coupling.

## Installation

If [available in Hex](https://hex.pm/docs/publish), the package can be installed by adding `ccm` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:ccm, "~> 0.1.0"}
  ]
end
```

## Key Features

* Time-delay embedding - Reconstructs the attractor from univariate time series
* Cross-mapping - Predicts one variable from another's reconstructed state space
* Convergence testing - Checks if prediction skill improves with library size
* Bidirectional analysis - Tests causality in both directions
* Bootstrap sampling - Improves statistical reliability

## Main Functions

* `CCM.new/3` - Creates a new CCM analysis structure
* `cross_map/2` - Performs unidirectional CCM analysis
* `bidirectional_ccm/1` - Tests causality in both directions

## Usage

```elixir
# Generate test data
{x_series, y_series} = CCMExample.generate_coupled_logistic_maps(300, 0.15)

# Perform analysis
ccm = CCM.new(x_series, y_series, embedding_dim: 3, tau: 1, num_samples: 50)
results = CCM.bidirectional_ccm(ccm)

# Run the example
CCMExample.run()
```

## How CCM Works

Correlation-based inference is widely used to deal with causal relationships. Unfortunately, these methods are not suitable to deal with nonlinear systems, since even a simple nonlinear process can produce "mirage correlations" [Sugihara et al. 2012] where variables appear to be correlated, but this correlation may vanish or even change sign over different time periods. Such transient correlations can produce the appearance of non-stationarity that can obscure any statistical association, and more importantly they can suggest that coupled variables are not causally related at all. Thus, in a linear system, just as "correlation does not imply causation", in a nonlinear system lack of correlation does not imply lack of causation.

Convergent Cross Mapping (CCM) [Sugihara et al., 2012] is a technique based on dynamical systems theory. Its main purpose is to assess causal relationship between variables. The general idea of CCM is reconstructing system attractors from time series data [Takens, 1981], [Abarbanel, 1996], [Sauer et al. 1991], [Deyle and Sugihara 2011].

CCM works by constructing a shadow manifold from time series data of variables, one for each variable. These shadow manifolds are approximations of the original attractors. According to Taken's theorem, when different variables are present in the same dynamical system, their shadow manifolds are diffeomorphic to the original attractor (there is a 1:1 mapping between them). It means that if a variable influences to another variable, the shadow manifold of the dependent variable can be used to estimate values of the independent variable. This estimate is the _cross-map_.

### Takens' idea

Briefly, the state of a dynamical system can be thought of as a location in a _state space_, whose coordinate axes are the relevant interacting variables. The system state changes and evolves in time according to the rules/equations that describe the system dynamics, and this traces out a trajectory. The collection of these time-series trajectories forms a geometric object known as an attractor manifold, which describes empirically how variables relate to each other in time.

Each variable can be thought of as a projection of the system state onto a particular coordinate axis. In other words, a time series is simply the projection of the motion of the system onto a particular axis, and recorded over time. As such, each time series contains information about the underlying system dynamics. In fact, Takens’ embedding theorem shows that each variable contains information about all the others, which allows systems to be studied from just a single time-series [Takens, 1981] by taking time-lag coordinates of the single variable as proxies for the other variables.

### Lag value

Time delay τ (lag) is a required parameter to create the embedding of the time series. Since we represent time by the index of elements in the time series, τ should be an integer between 1 and the maximum lag value specified in the 'max_lag' parameter, where 0 < max_lag < length of the time series. We use some heuristics and estimate it using the first minimum of mutual information [Kantz and Schreiber, 1997] between the time series and a shifted version of itself. Since it is a heuristic estimate, it will not be the best choice for every kind of application. To skip this estimation, the lag value can be given as a parameter.

### Embedding

A dynamical system is the tuple (M, f, T), where M is a manifold (in our case, an Euclidean space), T is time domain and f is an evolution rule t → f<sup>t</sup> (t ∈ T) such that f<sup>t</sup> is a diffeomorphism of the manifold to itself. In other terms, f(t) is a diffeomorphism, for every time t in the domain T [Wikipedia](https://en.wikipedia.org/wiki/Dynamical_system). We define this f function as a transformation to represent the temporal distance of data points as spatial distance of states.

According to Takens' theorem [Takens, 1981], we can predict a causal relationship between time series by analysing their shadow manifolds [Sugihara et al., 2012]. It means that we can estimate the original attractor by embedding the original 1D time series, using time-delayed surrogate copies of it [Sugihara et al., 1990]. Or, in a more formal way, we can construct an E-dimensional shadow manifold M<sub>x</sub> from the original one-dimensional time series X as follows:

M<sub>x</sub> = (x<sub>t</sub>, x<sub>t−τ</sub>, x<sub>t−2τ</sub>, ... x<sub>t−(E−1)τ</sub>)

#### Embedding dimensions

To ensure the embedding will be topologically correct, the value of E should be chosen carefully. Too low values cause crossings in the dynamics and too high values cause the dynamics unfold several times.
According to Whitney's strong embedding theorem [Whitney, 1992], E ≤ 2n for n = 1, 2. There are proofs for E ≤ 2n − 1 unless n is a power of 2 [Haefliger, Hirsch], [Wall].
Often a best guess is enough (using the value which gives the best forecast skill) to estimate its value.
Although closed-form function is not known to determine the best value of E for all integers, its value can be estimated using the False Nearest Neighbor algorithm [Abarbanel, 1996].

#### Separating chaos from noise

Separating chaos from noise in a 1D time series is almost impossible without additional transformations, but embeddings make it possible. Only by looking the evolution of states (in a plot), we can see the difference. If the plot exhibits some kind of a structure, it is a sign of a non-linear, but deterministic system. If the points are completely random, it is a stochastic noise. 

### Prediction

The estimation of causal relationship is based on the idea of information sharing. It basically means that if variable Y depends on variable X, Y should contain information about X. This information can be used to predict future states of Y based on prior states of X. [Granger, 1969].

Since we are about to estimate future state based on past states, we treat it as a regression problem.

First we need to split both embedded time series to training and test sets. Each training set describes past states and used to train a regression model. Test sets will be prediction targets and will be extracted from the end of each time series. Since temporal ordering is important, we do not shuffle the datasets before splitting them.

To find a cross-mapped estimate of x<sub>t</sub>|M<sub>y</sub> we need to identify the corresponding y<sub>t</sub> in M<sub>y</sub>. Since M<sub>y</sub> is diffeomorphic to M<sub>x</sub>, the nearest neighbors around y<sub>t</sub> can be used to estimate x<sub>t</sub>. To form a bounding simplex around an E-dimensional point, we need to find E+1 nearest neighbor. We use these points (y<sub>t<sub>1</sub></sub>, y<sub>t<sub>2</sub></sub>, ... y<sub>t<sub>E+1</sub></sub>) to estimate x<sub>t</sub> as follows:

x̂<sub>t</sub>|M<sub>y</sub> = ∑<sup>E+1</sup><sub>i=1</sub>(w<sub>i</sub> * x<sub>t<sub>i</sub></sub>)

where weight w<sub>i</sub> are exponentially weighted with the Euclidean-distance of y<sub>t</sub> and its nearest neighbors:
    
w<sub>i</sub> = u<sub>i</sub> / ∑<sup>E+1</sup><sub>j=1</sub>u<sub>j</sub>, u<sub>i</sub> = exp(−(||y<sub>t</sub> − y<sub>t<sub>i</sub></sub>|| / ||y<sub>t</sub> - y<sub>t<sub>1</sub></sub>||)

The above estimation will be repeated for L consecutive slices of the training set of X. The list of these consecutive slices of a time series is called library, where each slice is longer than the previous, and the increment size is constant. As we discussed before, estimation is a regression problem, so we need more than one data points, that's the reason why we use libraries.
We use these estimations to check convergence. In this case, convergence means that the estimates will improve as the library becomes larger, because the longer the library, the more precise the representation of the attractor, as the nearest neighbors of a point will be closer.

The above steps were performed on to find a cross-mapped estimate of x<sub>t</sub>|M<sub>y</sub>, so they will be repeated to find the estimate of ŷ<sub>t</sub>|M<sub>x</sub> as well.

The next step is to calculate Pearson-correlation between L estimates of X and L values from X, and similarly, L estimates of Y and L values of Y. These correlations will be the indicators of the strength of causal relationship between X and Y.

The best indication of a causal relationship is that the Forecast Skill of one (unidirectional coupling) or both (bi-directional coupling) variables are _high_ and _converging_.

## Citations

* Haefliger, A., & Hirsch, M. (1963). On the existence and classification of differentiable embeddings. Topology, 2, 129-135.
* Granger, C. W. J. (1969). Investigating Causal Relations by Econometric Models and Cross-spectral Methods. Econometrica. 37 (3): 424–438. doi:10.2307/1912791. JSTOR 1912791.
* Takens F. (1981) Detecting strange attractors in turbulence. In: Rand D., Young LS. (eds) Dynamical Systems and Turbulence, Warwick 1980. Lecture Notes in * Sugihara, George; May M., Robert. (19 April 1990). Nonlinear forecasting as a way of distinguishing chaos from measurement error in time series. Nature, Vol. 344, No. 6268, pp. 734-741
* Sauer, T., Yorke, J. and Casdagli, M. (1991) Embedology. Journal of Statistical Physics, 65, 579. doi:10.1007/BF01053745
* Whitney, Hassler (1992), Eells, James; Toledo, Domingo (eds.), Collected Papers, Boston: Birkhäuser, ISBN 0-8176-3560-2
* Abarbanel, H.D.I. (1996) Analysis of Observed Chaotic Data. Springer-Verlag, New York, 272.
Mathematics, vol 898., doi:10.1007/BFb0091924
* Deyle ER, Sugihara G (2011) Generalized Theorems for Nonlinear State Space Reconstruction. PLoS ONE 6(3): e18295. doi:10.1371/journal.pone.0018295
* Sugihara, George; et al. (26 October 2012). Detecting Causality in Complex Ecosystems. Science. 338 (6106): 496–500., doi:10.1126/science.1227079