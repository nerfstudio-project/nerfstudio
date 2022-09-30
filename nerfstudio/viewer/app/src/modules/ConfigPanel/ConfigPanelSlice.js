// The function below is called a selector and allows us to select a value from
// the state. Selectors can also be defined inline where they're used instead of
// in the slice file. For example: `useSelector((state) => state.counter.value)`
export const selectTrainingState = (state) =>
  state.shared.rendering.training_state;
export const selectOutputOptions = (state) => state.shared.output_options;
export const selectColormapOptions = (state) => state.shared.colormap_options;
