import React, { useEffect } from 'react';
import { useColorMode } from '@docusaurus/theme-common';

export default function Root({ children }) {
  const { colorMode } = useColorMode();

  useEffect(() => {
    // Function to send theme to all iframes
    const sendThemeToIframes = () => {
      const iframes = document.querySelectorAll('iframe');
      iframes.forEach((iframe) => {
        try {
          iframe.contentWindow.postMessage(
            {
              type: 'themeChange',
              theme: colorMode,
            },
            '*'
          );
        } catch (e) {
          // Ignore errors
        }
      });
    };

    // Send theme when it changes
    sendThemeToIframes();

    // Also send theme on a slight delay to catch lazy-loaded iframes
    const timer = setTimeout(sendThemeToIframes, 100);

    // Listen for theme requests from iframes
    const handleMessage = (event) => {
      if (event.data && event.data.type === 'requestTheme') {
        event.source.postMessage(
          {
            type: 'themeChange',
            theme: colorMode,
          },
          '*'
        );
      }
    };

    window.addEventListener('message', handleMessage);

    return () => {
      clearTimeout(timer);
      window.removeEventListener('message', handleMessage);
    };
  }, [colorMode]);

  return <>{children}</>;
}

