import gradio as gr
import fitz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


documents = []
vectorizer = None
document_vectors = None


def process_pdf(pdf_file):
    global documents, vectorizer, document_vectors

    documents = []

    pdf = fitz.open(pdf_file)

    for page_number, page in enumerate(pdf):
        text = page.get_text()

        if text.strip():
            documents.append({
                "page": page_number + 1,
                "text": text
            })

    pdf.close()

    if not documents:
        return "No readable text was found in this PDF."

    vectorizer = TfidfVectorizer(stop_words="english")
    document_vectors = vectorizer.fit_transform(
        [doc["text"] for doc in documents]
    )

    return f"Document loaded successfully. {len(documents)} pages indexed."


def search_document(question):
    if not documents:
        return "Please upload a PDF first."

    question_vector = vectorizer.transform([question])

    similarities = cosine_similarity(
        question_vector,
        document_vectors
    ).flatten()

    best_indices = similarities.argsort()[-3:][::-1]

    results = []

    for index in best_indices:
        score = similarities[index]

        if score > 0:
            results.append(
                f"Page {documents[index]['page']} "
                f"(relevance: {score:.2f})\n\n"
                f"{documents[index]['text'][:1500]}"
            )

    if not results:
        return "I couldn't find relevant information in the document."

    return "\n\n---\n\n".join(results)


with gr.Blocks(title="AI Document Search") as app:

    gr.Markdown(
        """
        # 📚 AI-Powered Document Search

        Upload a PDF and ask questions about its contents.
        """
    )

    pdf_upload = gr.File(
        label="Upload PDF",
        file_types=[".pdf"]
    )

    status = gr.Textbox(
        label="Status"
    )

    question = gr.Textbox(
        label="Ask a question",
        placeholder="What is this document about?"
    )

    search_button = gr.Button("🔎 Search Document")

    answer = gr.Markdown(
        label="Search Results"
    )

    pdf_upload.change(
        process_pdf,
        inputs=pdf_upload,
        outputs=status
    )

    search_button.click(
        search_document,
        inputs=question,
        outputs=answer
    )


app.launch()