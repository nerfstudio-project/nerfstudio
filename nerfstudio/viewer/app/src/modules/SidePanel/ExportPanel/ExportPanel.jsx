/* eslint-disable react/jsx-props-no-spreading */
import * as React from 'react';

import { Box, Typography, Tab, Tabs } from '@mui/material';
import { LevaPanel, LevaStoreProvider, useCreateStore } from 'leva';
import BlurOnIcon from '@mui/icons-material/BlurOn';
import CategoryIcon from '@mui/icons-material/Category';
import PointcloudSubPanel from './PointcloudSubPanel';
import LevaTheme from '../../../themes/leva_theme.json';

interface TabPanelProps {
  children: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          <Typography>{children}</Typography>
        </Box>
      )}
    </div>
  );
}

function a11yProps(index: number) {
  return {
    id: `simple-tab-${index}`,
    'aria-controls': `simple-tabpanel-${index}`,
  };
}

export default function ExportPanel(props) {
  // unpack relevant information
  const sceneTree = props.sceneTree;
  const showExportBox = props.showExportBox;

  const [type_value, setTypeValue] = React.useState(0);

  const handleTypeChange = (event: React.SyntheticEvent, newValue: number) => {
    setTypeValue(newValue);
  };

  const pointcloudStore = useCreateStore();

  return (
    <div className="ExportPanel">
      <Tabs
        value={type_value}
        onChange={handleTypeChange}
        aria-label="export type"
        variant="fullWidth"
        centered
        sx={{ height: 55, minHeight: 55 }}
      >
        <Tab
          icon={<BlurOnIcon />}
          iconPosition="start"
          label="Point Cloud"
          disableRipple
          {...a11yProps(0)}
        />
        <Tab
          icon={<CategoryIcon />}
          iconPosition="start"
          label="Mesh"
          disableRipple
          {...a11yProps(1)}
        />
      </Tabs>
      <TabPanel value={type_value} index={0}>
        <LevaPanel
          store={pointcloudStore}
          className="Leva-panel"
          theme={LevaTheme}
          titleBar={false}
          fill
          flat
        />
        <LevaStoreProvider store={pointcloudStore}>
          <PointcloudSubPanel
            sceneTree={sceneTree}
            showExportBox={showExportBox}
          />
        </LevaStoreProvider>
      </TabPanel>
      <TabPanel value={type_value} index={1}>
        Coming Soon
      </TabPanel>
    </div>
  );
}
