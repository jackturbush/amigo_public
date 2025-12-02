import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import styles from './index.module.css';

export default function Home(): JSX.Element {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title}`}
      description="A friendly Python library for multidisciplinary design optimization on high-performance computing resources">
      <main className={styles.homeMain}>
        <div className={styles.titleContainer}>
          <Heading as="h1" className={styles.title}>
            Amigo
          </Heading>
        </div>

        <div className={styles.contentContainer}>
          <p>
            Amigo is a Python library for solving multidisciplinary analysis and optimization problems on high-performance 
            computing systems through automatically generated C++ wrappers. All application code is written in Python and 
            automatically compiled to C++ with automatic differentiation using A2D.
          </p>

          <p>
            Multiple backend implementations are supported: Serial, OpenMP, and MPI (CUDA for Nvidia GPUs is under development). 
            User code and model construction are independent of the target backend. Amigo integrates seamlessly with OpenMDAO 
            through <code>amigo.ExternalComponent</code> and can be used as a sub-optimization component with accurate 
            post-optimality derivatives.
          </p>

          <Heading as="h2" className={styles.sectionTitle}>
            Getting started
          </Heading>

          <p>
            To solve your first optimal control problem using <strong>Amigo</strong>, please check the{' '}
            <Link to="/docs/getting-started/introduction" className={styles.link}>documentation</Link>, or simply try our{' '}
            <Link to="/docs/tutorials/cart-pole" className={styles.link}>cart-pole tutorial</Link>.
          </p>
        </div>
      </main>
    </Layout>
  );
}

