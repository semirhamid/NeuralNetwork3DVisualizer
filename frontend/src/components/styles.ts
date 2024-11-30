// Styles
export const styles: { [key: string]: React.CSSProperties } = {
  container: { display: 'flex', height: '100vh', fontFamily: 'Arial, sans-serif' },
  sidebar: {
    width: '300px',
    padding: '20px',
    background: '#f4f4f4',
    borderRight: '1px solid #ddd',
    overflowY: 'auto',
  },
  list: { listStyle: 'none', padding: 0 },
  controls: {
    width: '300px',
    padding: '20px',
    background: '#ffffff',
    borderRight: '1px solid #ddd',
    display: 'flex',
    flexDirection: 'column',
    gap: '10px',
  },
  canvasContainer: { flex: 1, background: '#fefefe', position: 'relative' },
};