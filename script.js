// Featured projects data
const projects = [
    {
        title: "Deep-Drowsiness-Detection-using-YOLO",
        description: "Developed a real-time drowsiness detection system using YOLO, enhancing road safety through advanced computer vision techniques.",
        image: "https://via.placeholder.com/300x200.png?text=Drowsiness+Detection",
        link: "https://github.com/imanoop7/Deep-Drowsiness"
    },
    {
        title: "Fine-Tuning Llama-3 using Synthetic Data",
        description: "Implemented fine-tuning techniques on Llama-3 model using synthetic data, improving model performance for specific tasks.",
        image: "https://via.placeholder.com/300x200.png?text=Llama-3",
        link: "https://github.com/imanoop7/finetunnig-using-synthetic-data"
    },
    {
        title: "Self Corrective Coding Assistant",
        description: "Created an AI-powered coding assistant capable of self-correction, enhancing developer productivity and code quality.",
        image: "https://via.placeholder.com/300x200.png?text=Coding+Assistant",
        link: "https://github.com/imanoop7/self-corrective-codding-assistant"
    },
    {
        title: "CNN-from-Scratch",
        description: "A Convolutional Neural Network implemented from scratch using NumPy. This project includes a custom CNN architecture for recognizing handwritten digits from the MNIST dataset.",
        image: "https://via.placeholder.com/300x200.png?text=CNN",
        link: "https://github.com/imanoop7/CNN-from-Scratch"
    },
    {
        title: "UNet-from-Scratch-using-Python",
        description: "Implementation of the UNet architecture from scratch using Python and PyTorch, demonstrating deep learning concepts in image segmentation.",
        link: "https://github.com/imanoop7/UNet-from-Scratch-using-Python-and-Pytorch",
        image: "https://via.placeholder.com/300x200.png?text=UNET"
    }
];

// Professional experience data
const experience = [
    {
        company: "TCS",
        role: "Data Scientist & AI Engineer",
        duration: "October 2021 - Present",
        achievements: [
            "Led the development and deployment of three innovative GenAI projects on AWS and Azure, reducing customer support inquiries by 20% and increasing customer satisfaction by 15%.",
            "Built and deployed two AI/ML models into production on AWS Cloud, including a CNN image classification model with 90% accuracy and a web scraping solution automating data extraction from 100+ websites daily.",
            "Developed and tested over 20 GenAI Proof of Concepts (POCs), covering various model types and frameworks.",
            "Demonstrated proficiency in both AWS and Azure cloud environments, optimizing services and managing cloud architecture."
        ]
    }
];

// Medium articles data
const mediumArticles = [
    {
        title: "Running Ollama on Google Colab (Free Tier): A Step-by-Step Guide",
        description: "An Beginner Introduction, how to install Ollama on Google Colab.",
        link: "https://medium.com/towards-artificial-intelligence/running-ollama-on-google-colab-free-tier-a-step-by-step-guide-9ef74b1f8f7a",
        image: "https://via.placeholder.com/300x200.png?text=ollama"
    },
    {
        title: "Fine-tuning LLMs for Natural Language to SQL Query Generation Using Synthetic Data: A Comprehensive Guide for Beginners",
        description: "we’ll explore how to fine-tune Large Language Models (LLMs) to generate SQL queries from natural language inputs. This process, known as Natural Language to SQL (NL2SQL), is a powerful tool that allows non-technical users to interact with databases using everyday language. We’ll break down each step of the process, explaining key concepts and providing detailed instructions to help you understand and implement your own NL2SQL system.",
        link: "https://medium.com/towards-artificial-intelligence/fine-tuning-llms-for-natural-language-to-sql-query-generation-using-synthetic-data-a-comprehensive-38afdafc90b0",
        image: "https://via.placeholder.com/300x200.png?text=Synthetic+Data"
    },
    {
        title: "Fine-Tuning Phi-3 with Unsloth for Superior Performance on Custom Data",
        description: "Phi-3, a powerful large language model (LLM) from Microsoft AI, holds immense potential for various tasks. But to truly unlock its potential for your specific needs, fine-tuning on your custom data is crucial. This article delves into using Unsloth, a cutting-edge library, to streamline the fine-tuning process of Phi-3 for your unique dataset",
        link: "https://medium.com/towards-artificial-intelligence/fine-tuning-phi-3-with-unsloth-for-superior-performance-on-custom-data-2c14b3c1e90b",
        image: "https://via.placeholder.com/300x200.png?text=Unsloth"
    },
    {
        title: "Supercharge Your Workflow: Unlocking Advanced Concepts of Microsoft Autogen",
        description: "Autogen — a powerful AI framework designed to enhance productivity through automation. Intrigued, he begins with the basics, creating a simple assistant that can help draft code snippets and manage mundane tasks. But as he delves deeper, AK discovers a wealth of advanced features that promise to revolutionize his workflow.",
        link: "https://medium.com/towards-artificial-intelligence/supercharge-your-workflow-unlocking-advanced-concepts-of-microsoft-autogen-e572cb20078d",
        image: "https://via.placeholder.com/300x200.png?text=AGENT+AUTOGEN"
    },
    {
        title: "Building a SQL Agent Using CrewAI and Ollama: A Comprehensive Guide",
        description: "SQL (Structured Query Language) remains the backbone of database management and manipulation, allowing for precise data querying and retrieval. However, constructing and executing SQL queries, especially complex ones, can be a time-consuming and error-prone task. Enter the realm of AI-powered automation with CrewAI and Ollama, tools designed to streamline and enhance database operations through intelligent agents.This article will take you on a detailed journey to create a sophisticated SQL agent using CrewAI and Ollama. We will cover everything from the initial setup and database preparation to the implementation of SQL tools, defining agents, creating tasks, and executing the entire process. Whether you’re a database developer, data analyst, or a technical enthusiast, this guide will equip you with the knowledge and tools to build an efficient SQL agent.",
        link: "https://medium.com/@mauryaanoop3/building-a-sql-agent-using-crewai-and-ollama-a-comprehensive-guide-1ad089610056",
        image: "https://via.placeholder.com/300x200.png?text=CREWAI+Ollama"
    }
];

// Function to create project cards
function createProjectCards() {
    const projectGrid = document.getElementById('project-grid');
    projectGrid.innerHTML = ''; // Clear existing content
    projects.forEach(project => {
        const card = document.createElement('div');
        card.className = 'project-card';
        card.innerHTML = `
            <img src="${project.image}" alt="${project.title}">
            <h3>${project.title}</h3>
            <p>${project.description}</p>
        `;
        card.addEventListener('click', () => {
            window.open(project.link, '_blank');
        });
        projectGrid.appendChild(card);
    });
}

// Function to create experience section
function createExperienceSection() {
    const experienceSection = document.getElementById('experience');
    // Clear existing content to prevent duplication
    experienceSection.innerHTML = '<h2>Professional Experience</h2>';
    
    const experienceContainer = document.createElement('div');
    experienceContainer.className = 'experience-container';

    experience.forEach(job => {
        const jobDiv = document.createElement('div');
        jobDiv.className = 'job';
        jobDiv.innerHTML = `
            <h3>${job.company} - ${job.role}</h3>
            <p>${job.duration}</p>
            <ul>
                ${job.achievements.map(achievement => `<li>${achievement}</li>`).join('')}
            </ul>
        `;
        experienceContainer.appendChild(jobDiv);
    });

    experienceSection.appendChild(experienceContainer);
}

// Function to create Medium article cards
function createMediumArticleCards() {
    const articleGrid = document.getElementById('article-grid');
    articleGrid.innerHTML = ''; // Clear existing content
    mediumArticles.forEach(article => {
        const card = document.createElement('div');
        card.className = 'article-card';
        card.innerHTML = `
            <img src="${article.image}" alt="${article.title}">
            <h3>${article.title}</h3>
            <p>${article.description}</p>
        `;
        card.addEventListener('click', () => {
            window.open(article.link, '_blank');
        });
        articleGrid.appendChild(card);
    });
}

// Function to handle form submission
function handleFormSubmission(event) {
    event.preventDefault();
    const form = event.target;
    const name = form.elements.name.value;
    const email = form.elements.email.value;
    const message = form.elements.message.value;

    // Here you would typically send this data to a server
    console.log("Form submitted:", { name, email, message });
    alert("Thank you for your message! I'll get back to you soon.");
    form.reset();
}

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    createProjectCards();
    createExperienceSection();
    createMediumArticleCards();
    document.getElementById('contact-form').addEventListener('submit', handleFormSubmission);
});

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});

// Add this function at the end of the file
function typeWriter(text, i, fnCallback) {
    if (i < text.length) {
        document.getElementById("job-title").innerHTML = text.substring(0, i+1) + '<span aria-hidden="true"></span>';
        setTimeout(function() {
            typeWriter(text, i + 1, fnCallback)
        }, 100);
    } else if (typeof fnCallback == 'function') {
        setTimeout(fnCallback, 700);
    }
}

// Start the typewriter effect
document.addEventListener('DOMContentLoaded', function() {
    typeWriter("Data Scientist | AI Engineer", 0, function() {
        // typeWriter function callback
    });
    createProjectCards();
    createExperienceSection();
    createMediumArticleCards();
    createLatestItems();
    createUpcomingItems();
});

// Add this to your existing DOMContentLoaded event listener
window.onscroll = function() {scrollFunction()};

function scrollFunction() {
    if (document.body.scrollTop > 20 || document.documentElement.scrollTop > 20) {
        document.getElementById("scroll-to-top").style.display = "block";
    } else {
        document.getElementById("scroll-to-top").style.display = "none";
    }
}

document.getElementById("scroll-to-top").onclick = function() {
    document.body.scrollTop = 0; // For Safari
    document.documentElement.scrollTop = 0; // For Chrome, Firefox, IE and Opera
};

document.addEventListener('click', function(e) {
    if (e.target && e.target.classList.contains('cta-button')) {
        console.log('Button clicked:', e.target.href);
    }
});

const articles = [
    {
        title: "A Deep Dive into Retrieval-Augmented Generation (RAG) with HyDE: How to Enhance Your AI's Response Quality",
        description: "An in-depth exploration of Retrieval-Augmented Generation (RAG) using Hypothetical Document Embeddings (HyDE) to improve AI response quality.",
        link: "https://medium.com/@mauryaanoop3/a-deep-dive-into-retrieval-augmented-generation-rag-with-hyde-how-to-enhance-your-ais-response-4d7ac0b8c200",
        type: "Article"
    },
    {
        title: "Highlighting and Annotating PDFs on UI Using Streamlit for Retrieval-Augmented Generation (RAG)",
        description: "An article detailing how to build an AI-powered document assistant using Streamlit and RAG, focusing on PDF highlighting and annotation.",
        link: "https://medium.com/@mauryaanoop3/from-pdfs-to-answers-building-an-ai-powered-document-assistant-with-streamlit-and-rag-bb3cd9478937",
        type: "Article"
    },
    {
        title: "My Journey as a Beginner: Implementing Research Papers from Scratch",
        description: "An article detailing my experience and insights gained from implementing complex research papers as a beginner in implementing research papers from scratch.",
        link: "https://medium.com/@mauryaanoop3/my-journey-as-a-beginner-implementing-research-papers-from-scratch-15d88ba2a819",
        type: "Article"
    }
];

const latestItems = [
    {
        title: "Building Your Own Generative Search Engine for Local Files Using Open-Source Models 🧐📂",
        description: "A comprehensive guide on creating a generative search engine for local files using open-source models, FAISS, and sentence transformers.",
        link: "https://medium.com/towards-artificial-intelligence/building-your-own-generative-search-engine-for-local-files-using-open-source-models-b09af871751c",
        type: "Article"
    },
    {
        title: "Building Your Own Generative Search Engine for Local Files Using Open-Source Models 🧐📂:Part-2",
        description: "A sequel to the first article, focusing on enhancing the generative search engine with visual capabilities using the LLaVA model.",
        link: "https://medium.com/towards-artificial-intelligence/building-your-own-generative-search-engine-for-local-files-using-open-source-models-part-2-4e869f62a9ee",
        type: "Article"
    },
    {
        title: "Fine-Tuning BERT for Phishing URL Detection: A Beginner's Guide",
        description: "A guide on fine-tuning BERT for the specific task of phishing URL detection, covering essential concepts and providing a comprehensive example using Python.",
        link: "https://medium.com/towards-artificial-intelligence/fine-tuning-bert-for-phishing-url-detection-a-beginners-guide-619fad27db41",
        type: "Article"
    },
    {
        title: "GPT-2 Tokenizer from Scratch",
        description: "An implementation of the GPT-2 tokenizer from scratch, demonstrating the inner workings of this crucial component in natural language processing.",
        link: "https://github.com/imanoop7/GPT-2-Tokenizer-from-Scratch",
        type: "Project"
    },
    {
        title: "Generative Search Engine For Local Files",
        description: "An AI-powered document search and question-answering system that allows users to explore and extract information from their local documents using natural language queries.",
        link: "https://github.com/imanoop7/Generative-Search-Engine-For-Local-Files",
        type: "Project"
    }
];

function createLatestItems() {
    const container = document.getElementById('latest-content');
    container.innerHTML = ''; // Clear existing content
    latestItems.forEach(item => {
        const itemDiv = document.createElement('div');
        itemDiv.className = 'latest-item';
        itemDiv.innerHTML = `
            <h3>${item.title}</h3>
            <p>${item.description}</p>
            <a href="${item.link}" target="_blank" class="cta-button">${item.type === 'Project' ? 'View Project' : 'Read Article'}</a>
        `;
        container.appendChild(itemDiv);
    });
}

// Update the DOMContentLoaded event listener
document.addEventListener('DOMContentLoaded', function() {
    typeWriter("Data Scientist | AI Engineer", 0, function() {
        // typeWriter function callback
    });
    createProjectCards();
    createExperienceSection();
    createMediumArticleCards();
    createLatestItems();
});
