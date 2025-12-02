import React from 'react';
import Content from '@theme-original/DocSidebar/Desktop/Content';
import { useLocation } from '@docusaurus/router';
import styles from './styles.module.css';

export default function ContentWrapper(props) {
  const location = useLocation();
  
  return (
    <>
      {/* Search bar at top of sidebar */}
      <div className={styles.searchContainer}>
        <input
          type="text"
          placeholder="Search docs (Ctrl + /)"
          className={styles.searchInput}
        />
      </div>
      <Content {...props} />
    </>
  );
}

