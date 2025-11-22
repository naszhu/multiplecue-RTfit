# ==========================================================================
# Run Flags Module
# Central place for toggling plotting/display options for main scripts
# ==========================================================================

module RunFlags

# Import ModelConfig from the top-level Config module
using Main.Config: ModelConfig

export get_plot_config, SAVE_INDIVIDUAL_CONDITION_PLOTS

# Toggle whether to write individual condition plots (e.g., _condition_1, _condition_2)
# Combined plots will still be generated.
const SAVE_INDIVIDUAL_CONDITION_PLOTS = false

# Configure whether target/distractor choice lines appear in plots
get_plot_config() = ModelConfig(false, false)

end # module
