import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow the dev server to proxy API requests to the backend
  // (useful during development to avoid CORS issues)
  async rewrites() {
    const apiBase = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8100";
    return [
      {
        source: "/proxy/api/:path*",
        destination: `${apiBase}/api/:path*`,
      },
      {
        source: "/proxy/fs/:path*",
        destination: `${apiBase}/fs/:path*`,
      },
    ];
  },
};

export default nextConfig;
