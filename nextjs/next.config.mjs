/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    const apiBase = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8100";
    return [
      { source: "/api/:path*", destination: `${apiBase}/api/:path*` },
    ];
  },
};

export default nextConfig;
