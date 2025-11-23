# ==========================================================================
# Fitting Utilities Module
# Re-exports optimization, plotting, and results utilities
# ==========================================================================

module FittingUtils

# Include submodules
include("optimization_utils.jl")
include("plotting_utils.jl")
include("results_utils.jl")

# Import submodules
using .OptimizationUtils
using .PlottingUtils
using .ResultsUtils

# Re-export all functions from submodules
export fit_model
export save_results, save_results_dual, save_results_single, save_results_allconditions
export generate_plot, generate_plot_dual, generate_plot_single, generate_plot_allconditions
export generate_accuracy_plot_dual, generate_overall_accuracy_plot, generate_overall_accuracy_plot_single, generate_overall_accuracy_plot_allconditions
export mis_lba_dualmodes_loglike

include("plotting_utils_dualmodes.jl")
using .PlottingUtilsDualModes

end # module
