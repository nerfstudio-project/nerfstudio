import { useContext } from 'react';
import { ReactReduxContext } from 'react-redux';

export function subscribe_to_changes(selector_fn, callback_fn) {
  // eslint-disable-next-line react-hooks/rules-of-hooks
  const { store } = useContext(ReactReduxContext);

  let previousState;
  const handleChange = () => {
    const currentState = selector_fn(store.getState());
    if (previousState !== currentState) {
      callback_fn(previousState, currentState);
      previousState = currentState;
    }
  };

  store.subscribe(handleChange);
}
