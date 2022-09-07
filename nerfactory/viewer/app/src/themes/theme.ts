import { createTheme } from '@mui/material/styles';

export const appTheme = createTheme({
  palette: {
    primary: { main: '#EEEEEE' },
    secondary: { main: '#FFD369' },
    text: {
      primary: '#EEEEEE',
      secondary: '#FFD369',
    },
  },
  components: {
    MuiTextField: {
      styleOverrides: {
        root: {
          '& label': {
            color: '#999999',
          },
          '& label.Mui-focused': {
            color: '#FFD369',
          },
          '& .MuiOutlinedInput-root': {
            '& fieldset': {
              borderColor: '#555555',
            },
            '&:hover fieldset': {
              borderColor: '#dddddd',
            },
            '&.Mui-focused fieldset': {
              borderColor: '#FFD369',
            },
          },
        },
      },
    },
    MuiButtonBase: {
      defaultProps: {
        disableRipple: true, // No more ripple, on the whole application 💣!
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          '&.Mui-disabled': {
            color: '#999999',
            backgroundColor: '#393e46',
          },
        },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: {
          fontSize: '.8rem',
        },
      },
    },
    MuiDivider: {
      styleOverrides: {
        root: {
          backgroundColor: '#555555',
        },
      },
    },
  },
});
