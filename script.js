// Featured projects data
const projects = [
    {
        title: "Deep-Drowsiness-Detection-using-YOLO",
        description: "Developed a real-time drowsiness detection system using YOLO, enhancing road safety through advanced computer vision techniques.",
        image: "https://via.placeholder.com/300x200.png?text=Drowsiness+Detection",
        link: "https://github.com/imanoop7/Deep-Drowsiness-Detection-using-YOLO"
    },
    {
        title: "Fine-Tuning Llama-3 using Synthetic Data",
        description: "Implemented fine-tuning techniques on Llama-3 model using synthetic data, improving model performance for specific tasks.",
        image: "https://via.placeholder.com/300x200.png?text=Llama-3+Fine-Tuning",
        link: "https://github.com/imanoop7/llama-3-fine-tuning"
    },
    {
        title: "Self Corrective Coding Assistant",
        description: "Created an AI-powered coding assistant capable of self-correction, enhancing developer productivity and code quality.",
        image: "https://via.placeholder.com/300x200.png?text=Coding+Assistant",
        link: "https://github.com/imanoop7/self-corrective-coding-assistant"
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
        description: "Ollama empowers you to leverage powerful large language models (LLMs) like Llama2,Llama3,Phi3 etc. without needing a powerful local machine. Google Colab’s free tier provides a cloud environment perfectly suited for running these resource-intensive models. This guide meticulously details setting up and running Ollama on the free version of Google Colab, allowing you to explore the capabilities of LLMs without significant upfront costs.",
        link: "https://medium.com/towards-artificial-intelligence/running-ollama-on-google-colab-free-tier-a-step-by-step-guide-9ef74b1f8f7a",
        image: "https://via.placeholder.com/300x200.png?text=ollama"
    },
    {
        title: "Fine-tuning LLMs for Natural Language to SQL Query Generation Using Synthetic Data: A Comprehensive Guide for Beginners",
        description: "we’ll explore how to fine-tune Large Language Models (LLMs) to generate SQL queries from natural language inputs. This process, known as Natural Language to SQL (NL2SQL), is a powerful tool that allows non-technical users to interact with databases using everyday language. We’ll break down each step of the process, explaining key concepts and providing detailed instructions to help you understand and implement your own NL2SQL system.",
        link: "https://medium.com/towards-artificial-intelligence/fine-tuning-llms-for-natural-language-to-sql-query-generation-using-synthetic-data-a-comprehensive-38afdafc90b0",
        image: "https://via.placeholder.com/300x200.png?text=Fine+Tunning+Using+Synthetic+Data"
    },
    {
        title: "Fine-Tuning Phi-3 with Unsloth for Superior Performance on Custom Data",
        description: "Phi-3, a powerful large language model (LLM) from Microsoft AI, holds immense potential for various tasks. But to truly unlock its potential for your specific needs, fine-tuning on your custom data is crucial. This article delves into using Unsloth, a cutting-edge library, to streamline the fine-tuning process of Phi-3 for your unique dataset",
        link: "https://medium.com/towards-artificial-intelligence/fine-tuning-phi-3-with-unsloth-for-superior-performance-on-custom-data-2c14b3c1e90b",
        image: "https://via.placeholder.com/300x200.png?text=Fine+Tuning-Phi3+with+Unsloth"
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
        image: "https://via.placeholder.com/300x200.png?text=NLP+Deep+Dive"
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
        experienceSection.appendChild(jobDiv);
    });
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