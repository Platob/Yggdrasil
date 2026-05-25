import type { NextConfig } from "next";

const BOT_API_URL = process.env.BOT_API_URL || "http://127.0.0.1:8100";

const nextConfig: NextConfig = {
  poweredByHeader: false,
  compress: true,

  async rewrites() {
    return [
      {
        source: "/api/bot/:path*",
        destination: `${BOT_API_URL}/api/:path*`,
      },
    ];
  },

  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "X-Frame-Options", value: "DENY" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
        ],
      },
      {
        source: "/_next/static/(.*)",
        headers: [
          {
            key: "Cache-Control",
            value: "public, max-age=31536000, immutable",
          },
        ],
      },
      {
        source: "/api/config",
        headers: [
          {
            key: "Cache-Control",
            value: "public, max-age=300, s-maxage=600, stale-while-revalidate=60",
          },
        ],
      },
    ];
  },
};

export default nextConfig;
