import { createTheme } from '@mui/material/styles';

export const appTheme = createTheme({
  palette: {
    primary: { main: '#d1d4db' },
    secondary: { main: '#ffd65c' },
    text: {
      primary: '#d1d4db',
      secondary: '#ff8600',
    },
  },
  components: {
    MuiTextField: {
      styleOverrides: {
        root: {
          '& label': {
            color: '#606980',
          },
          '& label.Mui-focused': {
            color: '#ff8600',
          },
          '& .MuiOutlinedInput-root': {
            '& fieldset': {
              borderColor: '#373c4b',
            },
            '&:hover fieldset': {
              borderColor: '#606980',
              borderWidth: '0.15rem',
            },
            '&.Mui-focused fieldset': {
              borderColor: '#ff8600',
            },
          },
        },
      },
    },
  },
});
