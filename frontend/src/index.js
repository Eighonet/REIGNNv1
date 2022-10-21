import React from 'react';
import ReactDOM from 'react-dom';
import './css/index.css';
import App from './default_page/App';
import Main from './Main';
import reportWebVitals from './default_page/reportWebVitals';

ReactDOM.render(
  <React.StrictMode>
    <Main />
  </React.StrictMode>,
  document.getElementById('root')
);