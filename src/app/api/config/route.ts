const CONFIG_RESPONSE = Response.json(
  {
    version: "1.0.0",
    features: {
      chat: true,
      execute: true,
      remoteCall: true,
    },
  },
  {
    headers: {
      "Cache-Control": "public, max-age=300, s-maxage=600, stale-while-revalidate=60",
    },
  },
);

export async function GET() {
  return CONFIG_RESPONSE.clone();
}
