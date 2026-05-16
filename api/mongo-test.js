const { MongoClient } = require('mongodb');

function safeHost(uri) {
  if (!uri) return null;
  try {
    return new URL(uri).hostname;
  } catch {
    return null;
  }
}

module.exports = async (req, res) => {
  const requiredToken = process.env.MONGO_TEST_TOKEN;
  if (requiredToken) {
    const supplied = req.headers['x-mongo-test-token'] || req.query.token;
    if (supplied !== requiredToken) {
      return res.status(401).json({ ok: false, message: 'unauthorized' });
    }
  }

  const uri = process.env.MONGODB_URI;
  const payload = {
    ok: false,
    checked_at: new Date().toISOString(),
    uri_present: Boolean(uri),
    uri_host: safeHost(uri),
    message: null,
    topology_type: null,
    servers: [],
  };

  if (!uri) {
    payload.message = 'MONGODB_URI is not configured';
    return res.status(500).json(payload);
  }

  const client = new MongoClient(uri, { serverSelectionTimeoutMS: 10000 });

  try {
    await client.db('admin').command({ ping: 1 });
    const admin = client.db('admin');
    const buildInfo = await admin.command({ buildInfo: 1 });
    payload.ok = true;
    payload.message = 'ok';
    payload.version = String(buildInfo.version || 'unknown');
    return res.status(200).json(payload);
  } catch (error) {
    payload.message = error instanceof Error ? error.message : String(error);
    return res.status(503).json(payload);
  } finally {
    await client.close();
  }
};
