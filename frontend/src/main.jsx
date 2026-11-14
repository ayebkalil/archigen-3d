import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx';
import { ProjectProvider } from './state/ProjectContext.jsx';
import './index.css';
import './workspace.css';

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ProjectProvider><App /></ProjectProvider>
  </React.StrictMode>
);
