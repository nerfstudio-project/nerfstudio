import { useContext } from 'react';
import { ReactReduxContext } from 'react-redux';

export function subscribe_to_changes(selector_fn, fn) {
  // selector_fn: returns a value from the redux state
  // fn_valid: function to run on a valid input
  // fn_null: function to run on a null input
  const { store } = useContext(ReactReduxContext);

  let current;
  const handleChange = () => {
    const previous = current;
    current = selector_fn(store.getState());
    if (previous !== current) {
      fn(previous, current);
    }
  };
  store.subscribe(handleChange);
}
