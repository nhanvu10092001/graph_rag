const http = require('http');

const data = JSON.stringify({
  name: `Diagnostic Group ${Math.floor(Math.random() * 10000)}`
});

const options = {
  hostname: 'localhost',
  port: 8000,
  path: '/api/groups',
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Content-Length': data.length
  }
};

console.log("Sending POST /api/groups directly to backend...");
const req = http.request(options, res => {
  console.log(`STATUS: ${res.statusCode}`);
  console.log(`HEADERS: ${JSON.stringify(res.headers)}`);
  res.setEncoding('utf8');
  let body = '';
  res.on('data', chunk => body += chunk);
  res.on('end', () => {
    console.log(`BODY: ${body}`);
  });
});

req.on('error', e => {
  console.error(`Problem with request: ${e.message}`);
});

req.write(data);
req.end();
