# 🎬 Movie Sentiment Analyzer

AI-powered web application that classifies movie reviews as **Positive**, **Neutral**, or **Negative**.

## 🚀 Features

- Real-time sentiment analysis using machine learning
- Review submission form
- Dashboard for viewing sentiment trends
- REST API for predictions

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **ML:** Scikit-learn / HuggingFace Transformers
- **Database:** MongoDB Atlas
- **Frontend:** React, TailwindCSS
- **Deployment:** Render / Railway

## 📂 Project Structure

## 🏗️ Development Phases

- [x] Phase 1: Setup & Planning
- [ ] Phase 2: Model Training
- [ ] Phase 3: Backend API Development
- [ ] Phase 4: Frontend Development
- [ ] Phase 5: Database Integration
- [ ] Phase 6: Deployment

## 📋 Development Roadmap

### Phase 1: Setup & Planning ✅

- [x] GitHub repo created
- [x] Python environment configured
- [x] IMDb dataset loaded and explored
- [x] Project structure defined

### Phase 2: Model Development ✅

- [x] Preprocessed IMDb dataset (5000 training samples)
- [x] Trained Logistic Regression baseline (85-88% accuracy)
- [x] Trained Naive Bayes comparison model
- [x] Tested HuggingFace DistilBERT (90-93% accuracy)
- [x] Selected Logistic Regression for production (speed + accuracy balance)
- [x] Saved models to `backend/models/`
- [x] Created `model.py` inference wrapper
- [x] Verified prediction speed < 10ms per review

**Model Performance:**

- Accuracy: 85-88%
- F1 Score: 0.85+
- Avg Prediction Time: 1-5ms

## 📝 License

MIT License
