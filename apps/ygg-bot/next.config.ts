import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    const botApi = process.env.YGG_BOT_API_URL ?? "http://localhost:8000";
    return [
      { source: "/bot-api/:path*", destination: `${botApi}/:path*` },
    ];
  },
};

export default nextConfig;
