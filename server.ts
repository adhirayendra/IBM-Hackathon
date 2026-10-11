import express from 'express';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI, Type } from '@google/genai';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import fetch from 'node-fetch';

dotenv.config();

// ── MCP / Langflow helper ─────────────────────────────────────────────────────
const LF_BASE = process.env.LANGFLOW_BASE_URL || 'http://localhost:7860';
const LF_TOKEN = process.env.LANGFLOW_TOKEN || '';

/** POST to a Langflow flow endpoint and return parsed JSON */
async function callLangflow(flowId: string, inputValue: string, sessionId?: string) {
  const url = `${LF_BASE}/api/v1/run/${flowId}?stream=false`;
  const body: Record<string, any> = {
    input_value: inputValue,
    output_type: 'chat',
    input_type: 'chat',
  };
  if (sessionId) body.session_id = sessionId;

  const resp = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(LF_TOKEN ? { Authorization: `Bearer ${LF_TOKEN}` } : {}),
    },
    body: JSON.stringify(body),
  });

  if (!resp.ok) {
    const txt = await resp.text();
    throw new Error(`Langflow error ${resp.status}: ${txt.slice(0, 200)}`);
  }
  return resp.json();
}

/** Extract the assistant text from a Langflow run response */
function extractLangflowText(data: any): string {
  try {
    const outputs = data?.outputs?.[0]?.outputs?.[0];
    const text =
      outputs?.results?.message?.text ||
      outputs?.messages?.[0]?.message ||
      outputs?.messages?.[0]?.text ||
      outputs?.artifacts?.message ||
      data?.outputs?.[0]?.outputs?.[0]?.outputs?.text ||
      data?.result ||
      '';
    // Never return raw JSON — if nothing matched, return empty string
    return typeof text === 'string' ? text : '';
  } catch {
    return '';
  }
}
// ─────────────────────────────────────────────────────────────────────────────

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(express.json({ limit: '25mb' }));

const port = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;

// Initialize Gemini Client
const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
  httpOptions: {
    headers: {
      'User-Agent': 'aistudio-build',
    },
  },
});

/**
 * Helper to generate content with fallback models if the primary model fails or experiences high demand (e.g. 503).
 */
async function generateWithFallback(params: {
  contents: any;
  config?: any;
  preferredModels?: string[];
}) {
  const models = params.preferredModels || [
    'gemini-2.5-flash-lite',
    'gemini-2.5-flash',
    'gemini-3.8-flash',
  ];

  let lastError: any = null;
  for (const model of models) {
    try {
      console.log(`[Gemini] Attempting generation with model: ${model}`);
      const response = await ai.models.generateContent({
        model,
        contents: params.contents,
        config: params.config,
      });
      return response;
    } catch (err: any) {
      console.warn(`[Gemini] Model ${model} failed (${err?.message || err}). Trying fallback...`);
      lastError = err;
    }
  }
  throw lastError;
}

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    appName: 'Pemantau Gizi',
    hasApiKey: Boolean(process.env.GEMINI_API_KEY),
    timestamp: new Date().toISOString(),
  });
});

// Endpoint: AI Food Recognition & Portion Estimation
app.post('/api/analyze-food', async (req, res) => {
  try {
    const { imageBase64, mimeType = 'image/jpeg', foodPrompt, scanMode = 'food' } = req.body;

    if (!imageBase64 && !foodPrompt) {
      return res.status(400).json({ error: 'Foto makanan atau deskripsi makanan belum dipilih.' });
    }

    // If no API key configured or fallback requested
    if (!process.env.GEMINI_API_KEY) {
      return res.status(200).json({
        mock: true,
        foodName: foodPrompt || 'Mixed Berry & Fruit Salad Bowl',
        indonesianName: 'Mangkuk Buah Campur Segar & Beri',
        portionGrams: 280,
        portionDescription: '1 mangkuk sedang (280g) - Kiwi, Strawberry, Blueberry, Melon, Anggur',
        calories: 320,
        protein: 4.8,
        carbs: 68.2,
        fat: 2.1,
        fiber: 9.4,
        sugar: 44.0,
        sodium: 18,
        confidence: 94,
        foodCategory: 'Buah-buahan / Healthy Snack',
        ingredients: [
          { name: 'Kiwi & Buah Beri Campur', calories: 140, portion: '120g', colorTag: '#10b981' },
          { name: 'Melon & Anggur Hijau', calories: 110, portion: '110g', colorTag: '#06b6d4' },
          { name: 'Saus Yoghurt Madu Ringan', calories: 70, portion: '50g', colorTag: '#f59e0b' },
        ],
        healthInsights: 'Kaya akan antioksidan, vitamin C, dan serat tinggi. Sangat baik untuk pencernaan dan mengontrol gula darah.',
      });
    }

    const systemInstruction = `Anda adalah sistem AI ahli gizi dan visi komputer untuk aplikasi "Pemantau Gizi" di Indonesia.
Tugas Anda:
1. Menganalisis gambar makanan / deskripsi makanan dengan teliti.
2. Identifikasi nama makanan dalam Bahasa Indonesia dan umum internasional.
3. Estimasi porsi (gram dan deskripsi porsi wajar).
4. Hitung nilai gizi berdasarkan acuan Tabel Komposisi Pangan Indonesia (TKPI/DKBM Kementerian Kesehatan RI) atau USDA:
   - Kalori (kkal)
   - Protein (gram)
   - Karbohidrat (gram)
   - Lemak (gram)
   - Serat (gram)
   - Gula (gram)
   - Natrium / Sodium (mg)
5. Rincikan 2-4 bahan / komponen penyusun utama beserta estimasi kalorinya.
6. Berikan estimasi tingkat keyakinan (confidence: 0-100%).
7. Berikan rekomendasi nutrisi singkat (healthInsights).
Kembalikan respon DALAM FORMAT JSON SESUAI SCHEMA.`;

    const parts: any[] = [];
    if (imageBase64) {
      parts.push({
        inlineData: {
          mimeType,
          data: imageBase64.replace(/^data:image\/[a-z]+;base64,/, ''),
        },
      });
    }

    const promptText = foodPrompt
      ? `Analisis makanan ini: ${foodPrompt}. Berikan rincian komposisi gizi dan estimasi porsi standar.`
      : `Analisis makanan dalam foto ini secara akurat. Kenali jenis makanan, porsi, kalori, dan nutrisinya berdasarkan database gizi Indonesia / USDA.`;

    parts.push({ text: promptText });

    const response = await generateWithFallback({
      contents: { parts },
      config: {
        systemInstruction,
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            foodName: { type: Type.STRING, description: 'Nama makanan umum (Indonesia/Inggris)' },
            indonesianName: { type: Type.STRING, description: 'Nama lokal makanan Indonesia' },
            portionGrams: { type: Type.NUMBER, description: 'Estimasi berat porsi dalam gram' },
            portionDescription: { type: Type.STRING, description: 'Deskripsi porsi, mis. 1 piring sedang (~350g)' },
            calories: { type: Type.NUMBER, description: 'Total kalori (kkal)' },
            protein: { type: Type.NUMBER, description: 'Protein dalam gram' },
            carbs: { type: Type.NUMBER, description: 'Karbohidrat dalam gram' },
            fat: { type: Type.NUMBER, description: 'Lemak dalam gram' },
            fiber: { type: Type.NUMBER, description: 'Serat dalam gram' },
            sugar: { type: Type.NUMBER, description: 'Gula dalam gram' },
            sodium: { type: Type.NUMBER, description: 'Natrium dalam mg' },
            confidence: { type: Type.NUMBER, description: 'Confidence score persentase (1-100)' },
            foodCategory: { type: Type.STRING, description: 'Kategori makanan (Makanan Utama, Sarapan, Sayuran, Buah, Cemilan)' },
            ingredients: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  name: { type: Type.STRING },
                  calories: { type: Type.NUMBER },
                  portion: { type: Type.STRING },
                  colorTag: { type: Type.STRING, description: 'Hex color mis. #f87171, #38bdf8, #a3e635' },
                },
                required: ['name', 'calories'],
              },
            },
            healthInsights: { type: Type.STRING, description: 'Insight gizi ringkas 1-2 kalimat' },
          },
          required: ['foodName', 'calories', 'protein', 'carbs', 'fat', 'portionGrams'],
        },
      },
    });

    const rawJson = response.text ? response.text.trim() : '{}';
    const parsedData = JSON.parse(rawJson);
    return res.json(parsedData);
  } catch (error: any) {
    console.error('Error analyzing food:', error);
    // Graceful fallback jika semua model Gemini gagal/overloaded
    return res.status(200).json({
      mock: true,
      foodName: req.body?.foodPrompt || 'Menu Porsi Nutrisi Sehat',
      indonesianName: 'Estimasi Piring Sehat Seimbang',
      portionGrams: 300,
      portionDescription: '1 porsi piring sedang (~300g)',
      calories: 360,
      protein: 16.0,
      carbs: 46.0,
      fat: 11.5,
      fiber: 6.8,
      sugar: 5.2,
      sodium: 180,
      confidence: 82,
      foodCategory: 'Makanan Sehat',
      ingredients: [
        { name: 'Nasi / Karbohidrat Pokok', calories: 150, portion: '120g', colorTag: '#f59e0b' },
        { name: 'Lauk Berprotein', calories: 140, portion: '90g', colorTag: '#10b981' },
        { name: 'Sayuran / Serat', calories: 70, portion: '90g', colorTag: '#06b6d4' },
      ],
      healthInsights: 'Layanan AI utama sedang sibuk. Menampilkan perkiraan gizi referensi seimbang.',
    });
  }
});

// Endpoint: Recipe Recommendations to Close Nutrition Gap
app.post('/api/recommend-recipes', async (req, res) => {
  try {
    const {
      calorieGap = 650,
      proteinGap = 35,
      carbsGap = 60,
      fatGap = 15,
      fiberGap = 12,
      dietPreference = 'Seimbang',
      userGoals = 'Menjaga berat badan ideal & pemenuhan gizi optimal',
    } = req.body;

    if (!process.env.GEMINI_API_KEY) {
      // High-quality fallback data
      return res.json({
        gapSummary: {
          calorieGap,
          proteinGap,
          carbsGap,
          fatGap,
          fiberGap,
        },
        recipes: [
          {
            id: 'rec-1',
            title: 'Pepes Tahu Telur Jamur Kukus',
            subtitle: 'Kaya Protein Nabati & Rendah Lemak Jenuh',
            calories: 280,
            protein: 24,
            carbs: 18,
            fat: 10,
            fiber: 7,
            prepTime: '15 mnt',
            cookTime: '25 mnt',
            difficulty: 'Mudah',
            gapMatch: 'Menutup 68% defisit protein & 58% serat hari ini',
            ingredients: [
              '300g tahu putih organik, haluskan',
              '2 butir telur ayam',
              '100g jamur tiram, suwir halus',
              '1 ikat kemangi segar',
              'Bumbu halus: 4 bawang merah, 2 bawang putih, kunyit, kemiri',
              'Daun pisang pembungkus',
            ],
            instructions: [
              'Campurkan tahu halus dengan telur kocok dan bumbu rempah aromatik.',
              'Masukkan jamur tiram suwir dan daun kemangi, aduk hingga rata.',
              'Bungkus dengan daun pisang membentuk lontong pipih, sematkan lidi.',
              'Kukus selama 20-25 menit hingga matang sempurna dan wangi meresap.',
            ],
            tag: 'Tinggi Protein',
          },
          {
            id: 'rec-2',
            title: 'Tumis Brokoli Ayam Panggang Wijen',
            subtitle: 'Serat Tinggi & Antioksidan Kuat',
            calories: 340,
            protein: 32,
            carbs: 22,
            fat: 12,
            fiber: 8,
            prepTime: '10 mnt',
            cookTime: '15 mnt',
            difficulty: 'Mudah',
            gapMatch: 'Menutup 90% kekurangan protein & serat',
            ingredients: [
              '150g dada ayam fillet tanpa kulit, potong dadu',
              '200g kuntum brokoli hijau segar',
              '1 sdm minyak wijen & minyak zaitun',
              '2 siung bawang putih cincang & 1 ruas jahe',
              '1 sdt kecap asin rendah natrium',
            ],
            instructions: [
              'Rebus brokoli sebentar dalam air mendidih (blanching 1 menit) lalu tiriskan.',
              'Tumis bawang putih dan jahe dengan minyak wijen hingga harum.',
              'Masukkan potongan dada ayam, masak hingga berubah warna dan matang merata.',
              'Masukkan brokoli rebus dan bumbu kecap rendah natrium, aduk cepat lalu angkat.',
            ],
            tag: 'Tinggi Serat & Protein',
          },
          {
            id: 'rec-3',
            title: 'Smoothie Bowl Bayam Tropis & Biji Chia',
            subtitle: 'Superfood Penutup Defisit Mikronutrien',
            calories: 220,
            protein: 11,
            carbs: 36,
            fat: 4.5,
            fiber: 9.5,
            prepTime: '8 mnt',
            cookTime: '0 mnt',
            difficulty: 'Sangat Mudah',
            gapMatch: 'Menutup 80% defisit serat harian seketika',
            ingredients: [
              '1 genggam bayam hijau muda (baby spinach)',
              '1 buah pisang ambon beku',
              '100g nanas manis segar',
              '150ml susu kedelai atau susu rendah lemak',
              '1 sdm biji chia (chia seeds) & potongan kiwi untuk topping',
            ],
            instructions: [
              'Blender bayam, pisang beku, nanas, dan susu hingga bertekstur kental lembut.',
              'Tuangkan ke dalam mangkuk saji.',
              'Beri taburan biji chia dan irisan buah kiwi segar di atasnya.',
            ],
            tag: 'Kaya Serat',
          },
        ],
      });
    }

    const systemInstruction = `Anda adalah konsultan gizi dan chef kuliner sehat Indonesia.
Tugas Anda adalah merekomendasikan resep masakan yang dirancang KHUSUS untuk MENUTUP KEKURANGAN GIZI (Nutritional Gap) pengguna hari ini.
Kekurangan nutrisi pengguna:
- Defisit Kalori: ${calorieGap} kkal
- Defisit Protein: ${proteinGap} g
- Defisit Karbohidrat: ${carbsGap} g
- Defisit Lemak: ${fatGap} g
- Defisit Serat: ${fiberGap} g
Preferensi: ${dietPreference}.
Tujuan: ${userGoals}.

Buat 3 resep makanan lezat, praktis, berbasis bahan lokal Indonesia atau bahan sehat mudah dicari yang pas menutup defisit tersebut. Format JSON sesuai schema.`;

    const response = await generateWithFallback({
      contents: 'Rekomendasikan 3 resep masakan penutup gap gizi harian saya.',
      config: {
        systemInstruction,
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            gapAnalysis: { type: Type.STRING, description: 'Analisis ringkas defisit nutrisi user hari ini' },
            recipes: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  id: { type: Type.STRING },
                  title: { type: Type.STRING, description: 'Nama resep' },
                  subtitle: { type: Type.STRING, description: 'Manfaat spesifik resep' },
                  calories: { type: Type.NUMBER },
                  protein: { type: Type.NUMBER },
                  carbs: { type: Type.NUMBER },
                  fat: { type: Type.NUMBER },
                  fiber: { type: Type.NUMBER },
                  prepTime: { type: Type.STRING },
                  cookTime: { type: Type.STRING },
                  difficulty: { type: Type.STRING },
                  gapMatch: { type: Type.STRING, description: 'Bagaimana resep ini menutup defisit user' },
                  ingredients: { type: Type.ARRAY, items: { type: Type.STRING } },
                  instructions: { type: Type.ARRAY, items: { type: Type.STRING } },
                  tag: { type: Type.STRING },
                },
                required: ['id', 'title', 'calories', 'protein', 'carbs', 'fat', 'fiber', 'ingredients', 'instructions'],
              },
            },
          },
          required: ['recipes'],
        },
      },
    });

    const parsed = JSON.parse(response.text?.trim() || '{}');
    return res.json(parsed);
  } catch (error: any) {
    console.error('Error generating recipes:', error);
    return res.json({
      gapSummary: {
        calorieGap: req.body?.calorieGap || 650,
        proteinGap: req.body?.proteinGap || 35,
        carbsGap: req.body?.carbsGap || 60,
        fatGap: req.body?.fatGap || 15,
        fiberGap: req.body?.fiberGap || 12,
      },
      recipes: [
        {
          id: 'rec-fallback-1',
          title: 'Pepes Tahu Telur Jamur Kukus',
          subtitle: 'Kaya Protein Nabati & Rendah Lemak Jenuh',
          calories: 280,
          protein: 24,
          carbs: 18,
          fat: 10,
          fiber: 7,
          prepTime: '15 mnt',
          cookTime: '25 mnt',
          difficulty: 'Mudah',
          gapMatch: 'Menutup defisit protein & serat harian',
          ingredients: ['300g tahu putih organik', '2 butir telur ayam', '100g jamur tiram', '1 ikat kemangi segar'],
          instructions: ['Campurkan tahu dengan bumbu dan telur', 'Bungkus daun pisang', 'Kukus 20-25 menit'],
          tag: 'Tinggi Protein',
        },
      ],
    });
  }
});

// ── MCP-backed endpoints ──────────────────────────────────────────────────────

// Flow IDs (override via env vars if needed)
const FLOW_MENGUKUR_GIZI = process.env.FLOW_MENGUKUR_GIZI || 'mengukur-gizi';
const FLOW_IMAGE_GIZI = process.env.FLOW_IMAGE_GIZI || 'f7a2f956-4b7f-4693-b43d-90ccb566ef10';

/**
 * POST /api/mengukur-gizi
 * Body: { query: string, sessionId?: string }
 * Calls the "Mengukur Gizi" Langflow flow and returns the AI response text.
 */
app.post('/api/mengukur-gizi', async (req, res) => {
  try {
    const { query, sessionId } = req.body || {};
    if (!query) return res.status(400).json({ error: 'query wajib diisi' });

    const data = await callLangflow(FLOW_MENGUKUR_GIZI, query, sessionId);
    const text = extractLangflowText(data);
    return res.json({ text, raw: data });
  } catch (err: any) {
    console.error('[mengukur-gizi]', err.message);
    return res.status(500).json({ error: err.message });
  }
});

/**
 * POST /api/image-gizi
 * Body: { imageBase64: string, mimeType?: string, sessionId?: string }
 * Uploads the image to the Langflow "Flow Analisa Gizi Berbasis Foto" flow
 * and returns the nutritional analysis as text.
 */
app.post('/api/image-gizi', async (req, res) => {
  try {
    const { imageBase64, mimeType = 'image/jpeg', sessionId } = req.body || {};
    if (!imageBase64) return res.status(400).json({ error: 'imageBase64 wajib diisi' });

    // Strip data-URL prefix if present
    const base64Data = imageBase64.replace(/^data:image\/[a-z]+;base64,/, '');

    // 1. Upload file to Langflow
    const buffer = Buffer.from(base64Data, 'base64');
    const FormData = (await import('form-data')).default;
    const form = new FormData();
    form.append('file', buffer, { filename: 'food.jpg', contentType: mimeType });

    const uploadUrl = `${LF_BASE}/api/v1/files/upload/${FLOW_IMAGE_GIZI}`;
    const uploadResp = await fetch(uploadUrl, {
      method: 'POST',
      headers: {
        ...form.getHeaders(),
        ...(LF_TOKEN ? { Authorization: `Bearer ${LF_TOKEN}` } : {}),
      },
      body: form as any,
    });

    if (!uploadResp.ok) {
      const txt = await uploadResp.text();
      throw new Error(`Upload gagal ${uploadResp.status}: ${txt.slice(0, 200)}`);
    }

    const { file_path: filePath } = (await uploadResp.json()) as { file_path: string };

    // 2. Run flow with the uploaded file path
    const runUrl = `${LF_BASE}/api/v1/run/${FLOW_IMAGE_GIZI}?stream=false`;
    const runBody: Record<string, any> = {
      input_value: 'Analisis kandungan gizi makanan dalam foto ini.',
      output_type: 'chat',
      input_type: 'chat',
      tweaks: {
        // Pass the uploaded image to the ChatInput component
        ChatInput: { files: filePath },
      },
    };
    if (sessionId) runBody.session_id = sessionId;

    const runResp = await fetch(runUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(LF_TOKEN ? { Authorization: `Bearer ${LF_TOKEN}` } : {}),
      },
      body: JSON.stringify(runBody),
    });

    if (!runResp.ok) {
      const txt = await runResp.text();
      throw new Error(`Run flow gagal ${runResp.status}: ${txt.slice(0, 200)}`);
    }

    const data = await runResp.json();
    const text = extractLangflowText(data);
    return res.json({ text, filePath, raw: data });
  } catch (err: any) {
    console.error('[image-gizi]', err.message);
    return res.status(500).json({ error: err.message });
  }
});

// ─────────────────────────────────────────────────────────────────────────────

// Setup Vite middlewares in development
async function startServer() {
  if (process.env.NODE_ENV === 'production') {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (req, res) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  } else {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  }

  app.listen(port, '0.0.0.0', () => {
    console.log(`Pemantau Gizi server running at http://0.0.0.0:${port}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
});
