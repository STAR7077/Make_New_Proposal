# Proposal Generator API

A Node.js/Express backend to store sample proposals and generate new job proposals using DeepSeek.

## Usage

1. Install dependencies:
   
   ```bash
   npm install
   ```
2. Run the API:
   
   ```bash
   npm run dev
   ```
3. Use the endpoints to upload sample proposals and generate new ones.

---

## Frontend
- Served at `/app` when the server is running.
- Open `http://127.0.0.1:8000/app` in your browser.

---

## Environment Variables
Create a `.env` file in the project root:

```
DEEPSEEK_API_KEY=your-deepseek-key
# Optional model override
# DEEPSEEK_MODEL=deepseek-chat
```

---

## Health Check
- API: `GET /proposals` should return JSON array.
- Frontend: `GET /app` should serve the UI.

---

The `/generate` endpoint selects the top three sample proposals (via TF-IDF similarity) and asks DeepSeek to create three new proposals in those styles.
