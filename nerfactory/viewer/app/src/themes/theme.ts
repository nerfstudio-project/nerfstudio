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
    MuiButton: {
      defaultProps: {
        disableRipple: true,
      },
      styleOverrides: {
        root: {
          '&.Mui-disabled': {
            color: '#999999',
            backgroundColor: '#393e46',
          },
        },
      },
    },
    MuiIconButton: {
      styleOverrides: {
        root: {
          color: '#eeeeee',
          backgroundColor: '#393e46',
          '&:hover': {
            color: '#FFD369',
            backgroundColor: '#555555',
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
    MuiFilledInput: {
      styleOverrides: {
        root: {
          backgroundColor: '#393e46',
          '&:hover': {
            backgroundColor: '#555555',
          },
        },
      },
    },
    MuiInput: {
      styleOverrides: {
        root: {
          '&:before': {
            borderBottom: '2px solid #555555',
          },
          '&:after': {
            borderBottom: '2px solid #FFD369',
          },
        },
      },
    },
  },
});
