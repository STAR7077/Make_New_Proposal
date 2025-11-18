const express = require('express');
const path = require('path');
const fs = require('fs/promises');
const axios = require('axios');
const dotenv = require('dotenv');
const natural = require('natural');

dotenv.config();

const app = express();

const PORT = process.env.PORT || 8000;
const DEEPSEEK_API_KEY = process.env.DEEPSEEK_API_KEY || '';
const DEEPSEEK_MODEL = process.env.DEEPSEEK_MODEL || 'deepseek-chat';
const DEEPSEEK_API_URL = 'https://api.deepseek.com/chat/completions';

const DEFAULT_PROPOSALS = [
  {
    name: 'Question Proposal',
    content:
      "Hello,\n\nThank you for the opportunity. I’ve carefully reviewed your project — this is not just an AI integration, but the creation of a complete intelligent communication agent capable of handling calls, emails, and lead qualification with precision and natural interaction.\n\nBefore we begin, a few quick clarifications:\n●  Do you already use any telephony platform (Twilio, Asterisk, Dialogflow CX, etc.)?\n● Should the AI handle both voice and text interactions autonomously, or in a hybrid (AI + human handoff) model?\n● Do you have an existing CRM or sales pipeline system to integrate with?\n\nWith deep experience in AI agents, NLP, speech processing, and system integrations (voice + email + CRM), I can design and build a multi-channel AI assistant that automates communication, optimizes sales funnels, and ensures seamless collaboration between AI and human teams.\n\nI focus on delivering robust, scalable, and context-aware AI systems with voice synthesis, speech-to-text, intent recognition, and workflow automation — all tailored to your operational goals.\n\nI can start immediately and deliver an MVP that handles calls, email organization, and lead qualification — ready for real-world use and scaling."
  },
  {
    name: '2 Way suggestion Proposal',
    content:
      "Hi there,\nI just read all the requirements.\nHowever, I have carefully read your project description and submitted my bid after thorough research and preparation.\nThis is my proposal for project:\n\nThe objective is to build a professional landing page and integrated blog for your B2B IT consulting firm. The landing page will be designed to highlight services clearly, with a modern responsive layout optimized for lead conversion. The blog will be implemented with a CMS like WordPress to allow easy publishing of articles, industry updates, and news.\n\nI suggest two possible approaches:\nOption A: Develop the entire site on WordPress, including the landing page and blog. This ensures full CMS control, simple content updates, and access to powerful plugins for SEO and performance optimization.\nOption B: Create a custom-coded landing page (HTML, CSS, JavaScript, PHP) optimized for speed and conversions, and integrate it with a separate WordPress blog. This option offers maximum performance for the landing page while still giving you easy blog management.\n\nBoth approaches will ensure a modern, responsive design, SEO optimization, and smooth user experience aimed at generating leads and improving your online presence.\n\nLet's discuss in detail through chat.\n\nDeadline: 3 to 4 weeks\nBudget: 1,500 to 2,500 USD depending on chosen option and customization level"
  },
  {
    name: 'Robert - DSAT Homework Coach with OpenAI & Airtable Integration',
    content:
      "Hello,\n\nAs I am very strong in OpenAI API (Custom GPT Actions), Airtable API, and backend development with Node.js/Express and Python/FastAPI, I can confidently build your DSAT Homework Coach to enforce the step-by-step workflow and log all student activity (steps, timing, mastery) into Airtable for Softr dashboards.\n\nPlease check some of my related projects:\nhttps://github.com/CloudDev777/AI-Powered-Data-Enrichment-System\nhttps://github.com/CloudDev777/Serpapi_to_Airtable_Automation\nhttps://github.com/CloudDev777/AI-Content-Generation-Platform\n\nWith 7+ years of experience, I’ve delivered GPT-integrated education tools and data pipelines, including strict multi-step interactions, Airtable-first schemas, and lightweight backends that expose clean endpoints for client apps. I’ve implemented mastery tracking, spaced repetition, and adaptive difficulty tied to tagged content.\n\nI also focus on workflow architecture, validation gates at every step, exception handling with retries, structured logging/audit trails, idempotent updates, and rate limiting to ensure stability and scalability.\n\nI’m confident I can deliver an MVP in 1–2 weeks: a sandbox Airtable base (Items, Students, StudentProgress, Submissions), two endpoints (get_next_question, grade_step), and a connected Custom GPT Action that strictly enforces the paraphrase → prediction → elimination → final sequence, plus clear documentation for your team to maintain and merge into the main LMS.\n\nI’m available full-time (40+ hours/week) and can start immediately. Looking forward to hearing from you soon.\nBest regards,\nRobert"
  }
];

const __dirnameSafe = __dirname;
const BASE_DIR = __dirnameSafe;
const FRONTEND_DIR = path.join(BASE_DIR, 'frontend');
const PROPOSAL_STORE = path.join(BASE_DIR, 'proposal_store.json');

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

async function loadProposals() {
  try {
    const data = await fs.readFile(PROPOSAL_STORE, 'utf-8');
    const parsed = JSON.parse(data);
    if (Array.isArray(parsed) && parsed.length >= 3) {
      return parsed;
    }
    return [...DEFAULT_PROPOSALS];
  } catch (err) {
    if (err.code === 'ENOENT') {
      return [...DEFAULT_PROPOSALS];
    }
    throw err;
  }
}

async function saveProposals(proposals) {
  await fs.writeFile(PROPOSAL_STORE, JSON.stringify(proposals, null, 2), 'utf-8');
}

function listTermsAsVector(tfidf, index) {
  const vector = {};
  tfidf.listTerms(index).forEach(({ term, tfidf: weight }) => {
    vector[term] = weight;
  });
  return vector;
}

function cosineSimilarity(vecA, vecB) {
  const terms = new Set([...Object.keys(vecA), ...Object.keys(vecB)]);
  let dot = 0;
  let magA = 0;
  let magB = 0;
  terms.forEach((term) => {
    const a = vecA[term] || 0;
    const b = vecB[term] || 0;
    dot += a * b;
    magA += a * a;
    magB += b * b;
  });
  if (!magA || !magB) return 0;
  return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}

function aiSelectTopThree(jobDesc, proposals) {
  if (proposals.length <= 3) {
    return proposals.slice(0, 3);
  }
  const tfidf = new natural.TfIdf();
  tfidf.addDocument(jobDesc);
  proposals.forEach((p) => tfidf.addDocument(p.content));
  const jobVector = listTermsAsVector(tfidf, 0);
  const ranked = proposals.map((proposal, idx) => {
    const vector = listTermsAsVector(tfidf, idx + 1);
    return { proposal, score: cosineSimilarity(jobVector, vector) };
  });
  ranked.sort((a, b) => b.score - a.score);
  return ranked.slice(0, 3).map((item) => item.proposal);
}

async function generateWithDeepseek(jobDesc, sample) {
  if (!DEEPSEEK_API_KEY) {
    return '(DeepSeek API key not set; returning placeholder text)';
  }
  const prompt = `CRITICAL: You MUST analyze the sample proposal below FIRST, then write a new proposal that EXACTLY mimics its structure and style. You can combine and merger some proposals so you make wonderful proposal

STEP 1 - Analyze the sample:
- Count paragraphs, headings, bullets if any
- Note greeting style (formal/informal) or absence
- Note sign-off style or absence
- Observe sentence length (short/medium/long)
- Observe tone (confident/technical/conversational)
- Count total word count

STEP 2 - Write your proposal:
- Match paragraph count EXACTLY from sample
- Match heading structure EXACTLY (same number, same order)
- Match bullet usage EXACTLY (same sections, similar count)
- Match greeting/sign-off style if present in sample
- Match sentence length patterns
- Match overall tone and voice
- Keep total length around 180-250 words
- Write NEW content for the job description (no verbatim copying)

LANGUAGE REQUIREMENTS:
- Use US English spelling and grammar ONLY (e.g., 'color' not 'colour', 'organize' not 'organise', 'center' not 'centre').
- Use US English conventions (e.g., periods inside quotes, US date format MM/DD/YYYY, US punctuation).
- Write in natural US English as a native US citizen would write.

REFERENCE SAMPLE PROPOSAL (analyze this first):
---
${sample.content}
---

JOB DESCRIPTION (write proposal for this):
---
${jobDesc}
---

Output ONLY the final proposal text, no analysis.`;

  const body = {
    model: DEEPSEEK_MODEL,
    messages: [
      {
        role: 'system',
        content:
          'You MUST analyze the sample proposal structure/style first, then write a new proposal that exactly matches those patterns. Use US English spelling, grammar, and conventions only. This is mandatory.'
      },
      { role: 'user', content: prompt }
    ],
    max_tokens: 400,
    temperature: 0.3
  };

  try {
    const response = await axios.post(DEEPSEEK_API_URL, body, {
      headers: {
        Authorization: `Bearer ${DEEPSEEK_API_KEY}`,
        'Content-Type': 'application/json'
      },
      timeout: 60000
    });
    const choices = response.data?.choices ?? [];
    const message = choices[0]?.message?.content;
    return message || '(Empty response)';
  } catch (error) {
    if (error.response) {
      return `(DeepSeek error ${error.response.status}) ${JSON.stringify(error.response.data)}`;
    }
    return `(DeepSeek request failed) ${error.message}`;
  }
}

app.post('/proposals', async (req, res, next) => {
  try {
    const { name, content } = req.body;
    if (!name || !content) {
      return res.status(400).json({ detail: 'Both name and content are required.' });
    }
    const proposals = await loadProposals();
    proposals.push({ name, content });
    await saveProposals(proposals);
    res.json({ message: 'Proposal uploaded successfully.' });
  } catch (err) {
    next(err);
  }
});

app.get('/proposals', async (req, res, next) => {
  try {
    const proposals = await loadProposals();
    res.json(proposals);
  } catch (err) {
    next(err);
  }
});

app.post('/generate', async (req, res, next) => {
  try {
    const jobDescription = req.body?.job_description;
    if (!jobDescription || !jobDescription.trim()) {
      return res.status(400).json({ detail: 'Job description is required.' });
    }
    const proposals = await loadProposals();
    if (proposals.length < 3) {
      return res.status(400).json({ detail: 'At least 3 sample proposals are required.' });
    }
    const topThree = aiSelectTopThree(jobDescription, proposals);
    const [sampleA, sampleB, sampleC] = topThree;
    const [deepseekA, deepseekB, deepseekC] = await Promise.all([
      generateWithDeepseek(jobDescription, sampleA),
      generateWithDeepseek(jobDescription, sampleB),
      generateWithDeepseek(jobDescription, sampleC)
    ]);
    res.json({
      matched_samples: [sampleA, sampleB, sampleC],
      results: [
        { provider: 'deepseek', sample_index: 0, text: deepseekA },
        { provider: 'deepseek', sample_index: 1, text: deepseekB },
        { provider: 'deepseek', sample_index: 2, text: deepseekC }
      ]
    });
  } catch (err) {
    next(err);
  }
});

app.get('/favicon.ico', (req, res) => res.sendStatus(204));

app.get('/.well-known/appspecific/com.chrome.devtools.json', (req, res) => res.sendStatus(204));

app.use('/app', express.static(FRONTEND_DIR));

app.use((err, req, res, next) => {
  console.error(err);
  res.status(500).json({ detail: 'Internal server error.' });
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});

