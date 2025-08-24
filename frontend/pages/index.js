import Chat from '@/components/Chat'

export default function Home({ timestamp }) {
  return (
    <div className="min-h-screen relative">
      
      
      <Chat timestamp={timestamp} />
    </div>
  )
}

// Option 3: Use getServerSideProps for dynamic content
export async function getServerSideProps() {
  return {
    props: {
      timestamp: new Date().toISOString(),
    },
  }
}