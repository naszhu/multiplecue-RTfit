# ==========================================================================
# Run Flags Module
# Central place for toggling plotting/display options for main scripts
# ==========================================================================

module RunFlags

# Re-export centralized flags from Config for backward compatibility
using Main.Config: get_plot_config, SAVE_INDIVIDUAL_CONDITION_PLOTS

export get_plot_config, SAVE_INDIVIDUAL_CONDITION_PLOTS

end # module
