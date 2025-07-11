# 2wrap.com Comprehensive RAG Chatbot

## Enhanced Features

This RAG system now provides **comprehensive scraping** of 2wrap.com with first-person responses as 2wrap.

### 🚀 Quick Start

1. **Start the server:**
```bash
python -m uvicorn app.main:app --reload
```

2. **Comprehensive Scraping (Recommended):**
```bash
# This scrapes ALL content from 2wrap.com (50+ pages)
curl -X POST "http://localhost:8000/process-2wrap-comprehensive" \
  -H "Content-Type: application/json" \
  -d '{"force_refresh": false}'
```

3. **Query the chatbot:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What car wrapping services do you offer?"}'
```

### 📊 API Endpoints

#### GET `/status`
Check system status and get usage instructions

#### POST `/process-2wrap-comprehensive`
**Comprehensive scraping** - Gets ALL content from 2wrap.com including:
- All services (wrapping, detailing, PPF, ceramic coating, tinting)
- Complete pricing information
- Color options and vinyl types
- Gallery and portfolio
- Company information and process
- Testimonials and reviews
- FAQ and blog content

**Request body:**
```json
{
  "force_refresh": false  // Set to true to clear existing data
}
```

#### POST `/query` 
Query the chatbot - **Responds in first person as 2wrap**

**Request body:**
```json
{
  "query": "What are your prices for car wrapping?"
}
```

**Example first-person response:**
> "We offer comprehensive car wrapping services starting at $2,500 for a full vehicle wrap. Our pricing depends on the vehicle size and vinyl type you choose. We use premium 3M and Avery Dennison materials..."

### 🎯 Key Improvements

1. **Comprehensive Content Coverage**
   - Scrapes 50+ pages instead of just 5
   - Hardcoded important URLs to ensure nothing is missed
   - Covers all services, pricing, colors, gallery, etc.

2. **First-Person Responses**
   - Chatbot responds as 2wrap directly ("We offer...", "Our services include...")
   - Professional but friendly tone
   - Maintains business personality

3. **Enhanced Accuracy**
   - Confidence scoring with 70% threshold
   - Source attribution for transparency
   - Service-specific metadata for better matching

4. **Better Performance**
   - Smart caching (24hr content, 6hr robots.txt)
   - Batch processing for large datasets
   - HTML structure-aware chunking

### 🔧 Configuration

The system automatically scrapes these key 2wrap.com pages:
- `/services`, `/pricing`, `/gallery`, `/portfolio`
- `/car-wrapping`, `/detailing`, `/paint-protection`
- `/ceramic-coating`, `/window-tinting`, `/colors`
- `/about`, `/contact`, `/process`, `/testimonials`
- And many more...

### 💡 Usage Tips

1. **First time setup:** Use `/process-2wrap-comprehensive` to get all content
2. **Regular updates:** Run comprehensive scraping weekly to keep content fresh
3. **Force refresh:** Use `"force_refresh": true` to completely rebuild the knowledge base
4. **Custom queries:** The system works best with specific questions about 2wrap's services

### 🎨 Example Interactions

**Customer:** "Do you do ceramic coating?"
**2wrap Bot:** "Yes! We offer professional ceramic coating services to protect your vehicle's paint. Our ceramic coatings provide long-lasting protection against UV rays, chemicals, and environmental contaminants..."

**Customer:** "What's your pricing for window tinting?"
**2wrap Bot:** "We provide window tinting services with pricing that varies based on your vehicle type and the tint level you prefer. Our professional team uses high-quality films that..."

The bot now responds naturally as 2wrap, providing specific information from your website content!
