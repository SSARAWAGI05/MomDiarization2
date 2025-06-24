import re
import numpy as np
from numpy.linalg import norm
from numpy import dot
from groq import Groq

client = Groq(api_key="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")

def remove_think_tags(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def chunk_text(text, max_words=150):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def cluster_meeting_transcript(transcript):
    sentences = re.split(r'(?<=[.!?]) +', transcript)
    chunk_size = 5
    step_size = 2
    chunks = [' '.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), step_size)]

    embeddings = [np.random.rand(384) for _ in chunks]
    similarities = [
        dot(embeddings[i], embeddings[i + 1]) / (norm(embeddings[i]) * norm(embeddings[i + 1]))
        for i in range(len(embeddings) - 1)
    ]

    smoothed = []
    window = 3
    for i in range(len(similarities)):
        start, end = max(0, i - window // 2), min(len(similarities), i + window // 2 + 1)
        smoothed.append(np.mean(similarities[start:end]))

    avg, std = np.mean(smoothed), np.std(smoothed)
    threshold = avg - 1.2 * std
    boundaries = [0] + [i + 1 for i, s in enumerate(smoothed) if s < threshold]

    clusters = []
    for i in range(1, len(boundaries)):
        clusters.append(' '.join(chunks[boundaries[i - 1]:boundaries[i]]))
    if boundaries[-1] < len(chunks):
        clusters.append(' '.join(chunks[boundaries[-1]:]))

    return clusters

def generate_summary(text, mode="detailed"):
    if mode == "detailed":
        prompt = f"""
Please summarize the following segment of a meeting transcript in a clear and detailed way.
Capture all relevant points and insights from the text only:\n\n{text}
"""
    else:
        prompt = f"""
Summarize the following meeting transcript in a concise, high-level way.
Extract only the key topics and insights based on the content:\n\n{text}
"""
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192"
        )
        return remove_think_tags(chat_completion.choices[0].message.content)
    except Exception as e:
        return f"Error contacting Groq API: {e}"

def generate_latex_mom(summary_text):
    prompt = f"""
You are a LaTeX expert. Create a fully compilable LaTeX document for "Minutes of Meeting" (MoM) based ONLY on the content below. Do NOT include empty sections. Do NOT fabricate any information.

# Instructions:
- Output ONLY LaTeX code.
- Use clean, professional formatting for business MoMs.
- Include sections ONLY if relevant and present in the summary.
- The document must compile successfully in Overleaf without errors.
- Don't include any LaTeX packages that are not necessary for the document.
- Don't include anything unnecessary before the title, like [10pt]article[margin=0.75in]; strictly nothing should come before the title. It should be a professional document of LaTeX code. Double check the LaTeX code and make sure it works.
- Output only the content of MoM. Don't include any other text or explanation—just the LaTeX code.
- Include every relevant point and make sure the MoM has all the content. Don't break the content in between.
- Double check and make sure the LaTeX code is correct and works properly in Overleaf or QuickLaTeX.

# Summary:
{summary_text}
"""
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192"
        )
        return remove_think_tags(chat_completion.choices[0].message.content)
    except Exception as e:
        return f"Error generating LaTeX: {e}"

def is_valid_latex(latex_code):
    required_sections = ["\\begin{document}", "\\end{document}"]
    if not all(section in latex_code for section in required_sections):
        return False
    if "Missing" in latex_code or "Error" in latex_code:
        return False
    if latex_code.count("\\begin") != latex_code.count("\\end"):
        return False
    return True

def main():
    try:
        with open("/teamspace/studios/this_studio/whisperX/audio2.txt", "r") as file:
            transcript = file.read()
    except Exception as e:
        print(f"Error reading transcript file: {e}")
        return

    if not transcript.strip():
        print("Transcript is empty.")
        return

    clusters = cluster_meeting_transcript(transcript)
    cluster_summaries = []

    for cluster in clusters:
        if len(cluster.split()) > 300:
            chunks = chunk_text(cluster)
            summaries = [generate_summary(c) for c in chunks]
            cluster_summaries.append(" ".join(summaries))
        else:
            cluster_summaries.append(generate_summary(cluster))

    combined_summary = " ".join(cluster_summaries)
    final_summary = generate_summary(combined_summary, mode="highlevel")

    # Retry LaTeX generation if validation fails
    max_attempts = 3
    for attempt in range(max_attempts):
        latex_code = generate_latex_mom(final_summary)
        if is_valid_latex(latex_code):
            break
        print(f"[Attempt {attempt + 1}] Invalid LaTeX generated. Retrying...")

    if not is_valid_latex(latex_code):
        print("Final LaTeX generation failed after multiple attempts.")
        return

    print("\n--- LaTeX Minutes of Meeting ---\n")
    print(latex_code)

    # Save LaTeX to specified path
    try:
        output_path = "/teamspace/studios/this_studio/whisperX/Latex/mom.tex"
        with open(output_path, "w") as f:
            f.write(latex_code)
        print(f"\nLaTeX saved successfully to {output_path}")
    except Exception as e:
        print(f"Error saving LaTeX file: {e}")

if __name__ == "__main__":
    main()
