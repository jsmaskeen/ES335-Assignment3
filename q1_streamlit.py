import streamlit as st
import torch
import torch.nn as nn
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import re

MODEL_DIR = Path("saved_models_q1")
MODEL_DIR2 = Path("saved_models_q1_2")


class MLPTextGenerator_(nn.Module):
    def __init__(self, ctx_window, vocab_size, embedding_dim, hidden_dim, activation):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * ctx_window, hidden_dim)
        self.act = activation
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x)
        embeds = embeds.view(x.shape[0], -1)
        out = self.fc1(embeds)
        out = self.act(out)
        out = self.fc2(out)
        return out


def generate_text(
    model, itos, stoi, ctx_window, device, prompt="", max_new_words=100, temperature=1.0
):
    if temperature <= 0:
        st.warning("Warning: Temperature must be > 0. Setting to 1.0.")
        temperature = 1.0
    model.eval()
    prompt_words = prompt.lower().split()
    context_indices = [stoi.get(word, 0) for word in prompt_words]
    context = context_indices[-ctx_window:]
    if len(context) < ctx_window:
        context = [0] * (ctx_window - len(context)) + context
    generated_words = []
    with torch.no_grad():
        for _ in range(max_new_words):
            context_tensor = torch.tensor(context).view(1, -1).to(device)
            logits = model(context_tensor)
            scaled_logits = logits / temperature
            distribution = torch.distributions.categorical.Categorical(
                logits=scaled_logits
            )
            next_token_idx = distribution.sample().item()
            word = itos.get(next_token_idx)
            if word is None or word == "." or next_token_idx == 0:
                break
            generated_words.append(word)
            context = context[1:] + [next_token_idx]
    model.train()
    full_text = prompt_words + generated_words
    return " ".join(full_text)


@st.cache_data
def get_vocab(model_dir):
    vocab_path = model_dir / "stoi_sherlock.json"
    if not vocab_path.exists():
        st.error(f"Vocabulary file not found! Expected at: {vocab_path}")
        st.stop()
    with open(vocab_path, "r") as f:
        stoi = json.load(f)
    itos = {v: k for k, v in stoi.items()}
    return stoi, itos


@st.cache_resource
def load_model(model_dir, embedding_dim, ctx_sz, activation_name, model_type):
    stoi, itos = get_vocab(model_dir)
    vocab_size = len(stoi)
    activation_obj = nn.ReLU() if activation_name == "ReLU" else nn.Tanh()
    hidden_dim = 768
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLPTextGenerator_(
        ctx_sz, vocab_size, embedding_dim, hidden_dim, activation_obj
    )
    model_path = (
        model_dir / f"e{embedding_dim}_c{ctx_sz}_a{activation_name}_{model_type}.pth"
    )
    if not model_path.exists():
        st.warning(f"Model file not found! Expected at: {model_path}")
        return None, None, None, None, None
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, stoi, itos, ctx_sz, device


def visualize_embeddings(model, stoi):
    word_groups = {
        "names": ["holmes", "watson", "adler", "lestrade"],
        "pronouns": ["he", "she", "his", "her", "they", "them", "you", "me"],
        "verbs": [
            "see",
            "saw",
            "go",
            "went",
            "think",
            "thought",
            "is",
            "was",
            "are",
            "were",
        ],
        "objects": ["pipe", "violin", "door", "window", "street", "house", "cab"],
        "concepts": ["death", "life", "love", "fear", "crime", "case", "mystery"],
        "synonyms": [
            "observe",
            "examine",
            "see",
            "look",
            "deduce",
            "infer",
            "crime",
            "case",
        ],
        "antonyms": [
            "good",
            "evil",
            "quick",
            "slow",
            "true",
            "false",
            "innocent",
            "guilty",
        ],
    }
    synonym_pairs = [
        ("observe", "thought"),
        ("deduce", "infer"),
        ("crime", "case"),
        ("examine", "mystery"),
    ]
    antonym_pairs = [
        ("good", "evil"),
        ("death", "life"),
        ("quick", "slow"),
        ("true", "false"),
        ("innocent", "guilty"),
    ]
    words_to_plot = []
    group_labels = []
    for group, words in word_groups.items():
        for word in words:
            if word in stoi and word not in words_to_plot:
                words_to_plot.append(word)
                group_labels.append(group)
    if not words_to_plot:
        st.warning("None of the specified words were found in the vocabulary.")
        return
    all_embeddings = model.embedding.weight.data.cpu().numpy()
    word_indices = [stoi[word] for word in words_to_plot]
    selected_embeddings = all_embeddings[word_indices]
    tsne = TSNE(
        n_components=2,
        random_state=42,
        init="pca",
        max_iter=3000,
    )
    embeddings_2d = tsne.fit_transform(selected_embeddings)
    word_to_coord = {word: embeddings_2d[i] for i, word in enumerate(words_to_plot)}
    fig, ax = plt.subplots(figsize=(20, 16))
    unique_groups = list(word_groups.keys())
    cmap = plt.get_cmap("tab10")
    group_to_color = {group: cmap(i) for i, group in enumerate(unique_groups)}
    for i, word in enumerate(words_to_plot):
        x, y = embeddings_2d[i]
        group = group_labels[i]
        color = group_to_color[group]
        ax.scatter(x, y, color=color, alpha=0.8, s=100)
        ax.annotate(word, (x, y), ha="center", va="bottom", fontsize=10)
    for w1, w2 in synonym_pairs:
        if w1 in word_to_coord and w2 in word_to_coord:
            coord1, coord2 = word_to_coord[w1], word_to_coord[w2]
            ax.plot(
                [coord1[0], coord2[0]],
                [coord1[1], coord2[1]],
                "g-",
                alpha=0.6,
                linewidth=1.5,
            )
    for w1, w2 in antonym_pairs:
        if w1 in word_to_coord and w2 in word_to_coord:
            coord1, coord2 = word_to_coord[w1], word_to_coord[w2]
            ax.plot(
                [coord1[0], coord2[0]],
                [coord1[1], coord2[1]],
                "r--",
                alpha=0.6,
                linewidth=1.5,
            )
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=group,
            markerfacecolor=group_to_color[group],
            markersize=12,
        )
        for group in unique_groups
    ]
    legend_elements.extend(
        [
            plt.Line2D([0], [0], color="green", lw=2, label="Synonym Pair"),
            plt.Line2D(
                [0], [0], color="red", lw=2, linestyle="--", label="Antonym Pair"
            ),
        ]
    )
    ax.legend(
        handles=legend_elements, title="Word Groups & Relations", fontsize="large"
    )
    ax.set_title("t-SNE Visualization of Word Embeddings", fontsize=16)
    st.pyplot(fig)


class MLPTextGenerator_Code(nn.Module):
    def __init__(
        self, vocab_size, embedding_dim, hidden_dim, context_window, activation_fn
    ):
        super(MLPTextGenerator_Code, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(context_window * embedding_dim, hidden_dim)
        self.activation = activation_fn
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x).view(x.shape[0], -1)
        out = self.activation(self.fc1(embedded))
        return self.fc2(out)

def tokenize_prompt_string(prompt_code):
    token_specification = [
        ('INCL', r'#include'),
        ('HEAD',   r'<[^>]+>'),
        ('PREP', r'#\s*(define|ifdef|ifndef|endif)'),
        ('KWRD', r'\b(auto|break|case|char|const|continue|default|do|double|else|enum|extern|'
                r'float|for|goto|if|inline|int|long|register|restrict|return|short|signed|sizeof|'
                r'static|struct|switch|typedef|union|unsigned|void|volatile|while)\b'),
        ('IDFR', r'\b[A-Za-z_][A-Za-z0-9_]*\b'),
        ('NMBR', r'\b\d+(\.\d+)?\b'),
        ('STRN', r'\"(?:\\.|[^\"\\\\])*\"'),
        ('OPRT', r'==|!=|<=|>=|->|&&|\|\||\+\+|--|[+\-*/%=&|<>!~^]'),
        ('DLMT', r'[;:,.\\[\\]\\(\\)\\{\\}]'),
        ('NEWL', r'\n'),
        ('WHSP', r'[ \t]+'),
        ('MTAN', r'.'),
    ]
    master_pattern = re.compile('|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_specification))
    tokens = []
    for match in master_pattern.finditer(prompt_code):
        token_type = match.lastgroup
        if token_type == 'NEWL':
            tokens.append('\\n')
        elif token_type == 'WHSP':
            tokens.append('\\s')
        elif token_type != 'MTAN':
            tokens.append(match.group())
    return tokens



def generate_code(model, itos, stoi, context_words, block_size, device, max_len=30):
    model.eval()
    unk_idx = stoi.get("<UNKN>", 0)
    context = [stoi.get(word, unk_idx) for word in context_words]
    if len(context) < block_size:
        context = [unk_idx] * (block_size - len(context)) + context
    generated_code = ""
    with torch.no_grad():
        for _ in range(max_len):
            x = torch.tensor(context[-block_size:]).unsqueeze(0).to(device)
            y_pred = model(x)
            probs = torch.nn.functional.softmax(y_pred, dim=1)
            ix = torch.multinomial(probs, num_samples=1).item()
            word = itos.get(ix, "<UNKN>")  
            if ix == unk_idx:
                break
            context.append(ix)
            if word == "\\n":
                generated_code += "\n"
            elif word == "\\s":
                generated_code += " "
            else:
                generated_code += word
    return generated_code


@st.cache_data
def get_vocab_code(model_dir):
    vocab_path = model_dir / "stoi_linux.json"
    if not vocab_path.exists():
        st.error(f"Vocabulary file not found! Expected at: {vocab_path}")
        st.stop()
    with open(vocab_path, "r") as f:
        stoi = json.load(f)
    itos = {v: k for k, v in stoi.items()}
    return stoi, itos


@st.cache_resource
def load_model_code(model_dir, embedding_dim, ctx_sz, activation_name, model_type):
    stoi, itos = get_vocab_code(model_dir)
    vocab_size = len(stoi)
    activation_obj = nn.ReLU() if activation_name == "ReLU" else nn.Tanh()
    hidden_dim = 768
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLPTextGenerator_Code(
        vocab_size, embedding_dim, hidden_dim, ctx_sz, activation_obj
    )
    model_path = (
        model_dir / f"e{embedding_dim}_c{ctx_sz}_a{activation_name}_{model_type}.pth"
    )
    if not model_path.exists():
        st.warning(f"Model file not found! Expected at: {model_path}")
        return None, None, None, None, None
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, stoi, itos, ctx_sz, device


def visualize_code_embeddings(model, stoi):
    code_groups = {
         "types": ["int", "char", "void", "struct", "union", "enum", "double", "float", "long", "short", "unsigned", "signed", "typedef"],
        "control": ["if", "else", "for", "while", "return", "break", "switch"],
        "operators": ["+", "-", "*", "/", "=", "->", "&", "|", "=="],
        "delimiters": ["(", ")", "{", "}", "[", "]", ";", ","],
        "preprocessor": ["#include", "#define", "#ifdef", "#endif"],
    }
    words_to_plot, group_labels = [], []
    for group, words in code_groups.items():
        for word in words:
            if word in stoi and word not in words_to_plot:
                words_to_plot.append(word)
                group_labels.append(group)
    if not words_to_plot:
        st.warning("None of the specified code tokens were found in the vocabulary.")
        return
    embeddings = model.embedding.weight.data.cpu().numpy()
    word_indices = [stoi[word] for word in words_to_plot]
    selected_embeddings = embeddings[word_indices]
    tsne = TSNE(
        n_components=2, random_state=42, perplexity=min(5, len(word_indices)-1)
    )
    embeddings_2d = tsne.fit_transform(selected_embeddings)
    fig, ax = plt.subplots(figsize=(16, 12))
    unique_groups = list(code_groups.keys())
    colors = plt.cm.get_cmap("tab10", len(unique_groups))
    group_to_color = {group: colors(i) for i, group in enumerate(unique_groups)}
    for i, word in enumerate(words_to_plot):
        x, y = embeddings_2d[i]
        group = group_labels[i]
        ax.scatter(x, y, color=group_to_color[group], s=100)
        ax.annotate(
            repr(word).strip("'"),
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=group,
            markerfacecolor=group_to_color[group],
            markersize=10,
        )
        for group in unique_groups
    ]
    ax.legend(handles=legend_elements, title="Token Groups")
    ax.set_title("t-SNE Visualization of C Code Token Embeddings")
    st.pyplot(fig)


st.set_page_config(layout="wide")
st.title("ES335 Assignment 3 Question 1")

tab1, tab2 = st.tabs(["Category 1: Sherlock", "Category 2: Linux Kernel Code"])

with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.header("Model Configuration")
        text_embedding_dim = st.selectbox(
            "Embedding Dimension", [32, 64], key="text_emb"
        )
        text_ctx_sz = st.selectbox("Context Size", [4, 8, 12], key="text_ctx")
        text_activation_name = st.selectbox(
            "Activation Function", ["ReLU", "Tanh"], key="text_act"
        )

        st.header("Generation Parameters")
        text_temperature = st.slider(
            "Temperature", 0.01, 3.0, 1.0, 0.01, key="text_temp"
        )
        text_max_new_words = st.number_input(
            "Max Words to Generate", 1, 500, 50, key="text_max_words"
        )
        text_seed = st.number_input("Random Seed", 0, value=42, key="text_seed")

    with col2:
        text_prompt = st.text_area(
            "Enter your prompt:", "sherlock holmes was", height=100, key="text_prompt"
        )
        if st.button("Generate Text", key="text_generate"):
            torch.manual_seed(text_seed)
            best_model, stoi, itos, ctx_window, device = load_model(
                MODEL_DIR, text_embedding_dim, text_ctx_sz, text_activation_name, "best"
            )
            final_model, _, _, _, _ = load_model(
                MODEL_DIR,
                text_embedding_dim,
                text_ctx_sz,
                text_activation_name,
                "final",
            )

            if best_model and final_model:
                st.success("Both models loaded.")
                with st.spinner("Generating..."):
                    best_text = generate_text(
                        best_model,
                        itos,
                        stoi,
                        ctx_window,
                        device,
                        text_prompt,
                        text_max_new_words,
                        text_temperature,
                    )
                    final_text = generate_text(
                        final_model,
                        itos,
                        stoi,
                        ctx_window,
                        device,
                        text_prompt,
                        text_max_new_words,
                        text_temperature,
                    )

                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.subheader("Best Model (Min Validation Error)")
                    st.markdown(f"> {best_text}")
                    with st.expander("Visualize Embeddings (Best Model)"):
                        visualize_embeddings(best_model, stoi)
                with res_col2:
                    st.subheader("Final Model (Min Training Error)")
                    st.markdown(f"> {final_text}")
                    with st.expander("Visualize Embeddings (Final Model)"):
                        visualize_embeddings(final_model, stoi)
            else:
                st.error("Could not generate text. A model file might be missing.")

with tab2:
    col1_code, col2_code = st.columns([1, 2])
    with col1_code:
        st.header("Model Configuration")
        code_embedding_dim = st.selectbox(
            "Embedding Dimension", [32, 64], key="code_emb"
        )
        code_ctx_sz = st.selectbox("Context Size", [4, 8, 12], key="code_ctx")
        code_activation_name = st.selectbox(
            "Activation Function", ["ReLU", "Tanh"], key="code_act"
        )

        st.header("Generation Parameters")
        code_max_len = st.number_input(
            "Max Tokens to Generate", 1, 500, 50, key="code_max_len"
        )
        code_seed = st.number_input("Random Seed", 0, value=42, key="code_seed")

    with col2_code:
        code_prompt = st.text_area(
            "Enter C code prompt:",
            "#include <stdio.h>\n\nint main() {\n   ",
            height=150,
            key="code_prompt",
        )
        if st.button("Generate Code", key="code_generate"):
            torch.manual_seed(code_seed)
            best_model, stoi, itos, ctx_window, device = load_model_code(
                MODEL_DIR2,
                code_embedding_dim,
                code_ctx_sz,
                code_activation_name,
                "best",
            )
            final_model, _, _, _, _ = load_model_code(
                MODEL_DIR2,
                code_embedding_dim,
                code_ctx_sz,
                code_activation_name,
                "final",
            )

            if best_model and final_model:
                st.success("Models loaded successfully.")
                with st.spinner("Generating code..."):
                    tokenized_prompt = tokenize_prompt_string(code_prompt)
                    best_code = generate_code(
                        best_model,
                        itos,
                        stoi,
                        tokenized_prompt,
                        ctx_window,
                        device,
                        code_max_len,
                    )
                    final_code = generate_code(
                        final_model,
                        itos,
                        stoi,
                        tokenized_prompt,
                        ctx_window,
                        device,
                        code_max_len,
                    )

                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.subheader("Best Model (Min Validation Error)")
                    st.code(code_prompt + best_code, language="c")
                    with st.expander("Visualize Code Token Embeddings"):
                        visualize_code_embeddings(best_model, stoi)
                with res_col2:
                    st.subheader("Final Model (Min Training Error)")
                    st.code(code_prompt + final_code, language="c")
                    with st.expander("Visualize Code Token Embeddings"):
                        visualize_code_embeddings(final_model, stoi)
            else:
                st.error(
                    "Could not generate code. The selected model file might be missing."
                )
