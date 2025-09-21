
---
title: "Building AI Agents from Scratch: Understanding the Core Components Behind the Magic"
date: 2025-09-21T12:34:00+00:00
tags: ["ai-agents", "llm", "python", "langraph", "autogen", "crewai", "tool-integration", "agent-architecture", "machine-learning"]
author: "Anoop Maurya"
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: false
description: "A hands-on deep dive into building AI agents from scratch to understand the core components (LLM + memory + tools + loops) that power autonomous systems, moving beyond frameworks to master the fundamentals"
canonicalURL: "https://imanoop7.github.io/posts/attention_transformer_blog/"
disableHLJS: false
hideSummary: false
searchHidden: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: true
ShowRssButtonInSectionTermList: true
UseHugoToc: true
---

## Introduction

As a data scientist and AI engineer, I've spent countless hours working with various agentic frameworks like LangGraph, AutoGen, and CrewAI. While these tools are incredibly powerfull and make our lives easier, I often found myself wondering: *what actually happens under the hood?* What makes an AI agent tick? How do all these pieces come together to create something that can think, plan, and act autonomously?

That curiosity led me down a fascinating rabbit hole, and eventually to building an AI agent completly from scratch - no frameworks, no abstractions, just pure Python and a deep dive into the fundamental building blocks that make agents work.

In today's rapidly evolving AI landscape, we're witnessing an explosion of agentic systems. These aren't just chatbots that respond to queries - they're autonomous entities capable of complex reasoning, planning, and executing multi-step tasks. But here's the thing: most of us are building on top of these sophisticated frameworks without truely understanding what's happening beneath the surface.

**Why does this matter?** Because when you understand the fundamentals, you become a better AI engineer. You can debug issues more effectively, optimize performance better, and most importantly, you can innovate beyond the constraints of existing frameworks.

So let's pull back the curtain and build an agent from the ground up. By the end of this post, you'll have a clear understanding of what an AI agent really is, why we need them, and how to build one yourself.

### What is an AI Agent?

Before we dive into the code, let's establish a clear understanding of what we mean by an "AI agent." 

An AI agent is fundamentally different from a simple language model. While an LLM can generate text based on prompts, an agent can:
- **Think** about problems systematically
- **Plan** multi-step solutions  
- **Use tools** to interact with the world
- **Remember** previous interactions and learnings
- **Iterate** until a goal is achieved

Think of it this way:
- **LLM**: "Tell me about the weather"
- **Agent**: "Check the weather, plan my day accordingly, set reminders, and book restaurant reservations if it's going to rain"

The agent doesn't just provide information - it takes action.

##W The Agent Equation: Two Ways to Think About It

Throughout my research and experimentation, I've found two complementary ways to conceptualize what an agent is:

### Formula 1: The Loop Perspective
```
agent = llm + memory + tools + while loop
```

This formula emphasizes the *iterative* nature of agents. The while loop is crucial - it's what allows the agent to keep working until the task is complete, rather than giving a single response.

### Formula 2: The Capability Perspective  
```
agent = llm(read+write) + planning + tools + (condition+loop)
```

This formula emphasizes the *capabilities* of agents. The LLM can both read (understand) and write (generate), planning provides structure, tools provide actions, and conditions with loops provide control flow.

Both formulas are correct - they just highlight different aspects of the same underlying system.

## Why Build Agents from Scratch?

You might be thinking: "Why re-invent the wheel? LangGraph and AutoGen work perfectly fine!" 

And you're absolutley right - for production systems, you should definitely use these battle-tested frameworks. But here's why building from scratch is invaluable:

### 1. **Deep Understanding**
When you build something yourself, you understand every component. This knowledge is incredibly valuable when debugging production issues or optimizing performance.

### 2. **Framework Independence** 
Understanding the fundamentals means you're not locked into any particular framework. You can evaluate new tools more effectively and migrate between them when needed.

### 3. **Innovation Opportunities**
The best innovations often come from understanding the basics deeply enough to see new possibilities. Many of today's popular frameworks started as someone's experiment in building agents from scratch.

### 4. **Debugging Superpowers**
When your production agent misbehaves, knowing what's happening under the hood makes troubleshooting infinitely easier.

## The Architecture: Building Blocks of Our Agent

Let me walk you through the core components we'll be building:

### Component 1: The LLM Foundation
Our agent starts with a simple LLM interface. Here's how we set up the basic communication:

```python
from ai import openai

def basic_completion():
    completion = openai.chat.completions.create({
        "messages": [
            {"role": "developer", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is your name?"},
        ],
        "model": "llama3.2:latest",
    })
    return completion.get("choices", [None])[0]
```

This is our foundation - a simple way to communicate with an LLM. But it's just a one-shot interaction. To make it agentic, we need more.

### Component 2: Adding Conditions and Judgment
The next step is teaching our agent to evaluate its own responses:

```python
def answer_with_judgment(question):
    # First: get an answer
    completion = openai.chat.completions.create({
        "messages": [{"role": "developer", "content": question}],
        "model": "llama3.2:latest",
    })
    answer = completion.get("choices", [{}])[0].get("message", {}).get("content")
    
    # Second: judge the completeness
    check = openai.chat.completions.create({
        "model": "llama3.2:latest",
        "messages": [{
            "role": "developer",
            "content": (
                "You are a judge. Is this answer complete?\n\n"
                f"Question: {question}\n\nAnswer: {answer}"
            ),
        }],
    })
    
    judge_response = check.get("choices", [{}])[0].get("message", {}).get("content", "")
    is_complete = "yes" in judge_response.lower() or "complete" in judge_response.lower()
    
    return answer, is_complete
```

Now our agent can evaluate its own work! This self-reflection capability is crucial for building reliable agents.

### Component 3: Tool Integration
This is where things get really interesting. Tools allow our agent to interact with the outside world:

```python
def complete_with_tools(args):
    """Run completion and automatically handle tool calls."""
    completion = openai.chat.completions.create(args)
    
    choice0 = completion.get("choices", [{}])[0]
    message = choice0.get("message", {})
    tool_calls = message.get("tool_calls")
    
    if tool_calls:
        # Execute each tool call
        args["messages"].append(message)
        
        for tool_call in tool_calls:
            fn_name = tool_call.get("function", {}).get("name")
            fn_args = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
            
            # Execute the tool
            result = tool_functions[fn_name](fn_args)
            
            # Add result back to conversation
            args["messages"].append({
                "role": "tool",
                "tool_call_id": tool_call.get("id", ""),
                "content": result,
            })
        
        # Continue the conversation
        return complete_with_tools(args)
    
    return completion
```

This recusive approach allows our agent to use tools as needed and continue the conversation based on the results.

### Component 4: Planning and Memory
The final piece is giving our agent the ability to plan and remember:

```python
# Todo list functions for planning
_todos = []
_done = []

def add_todos(params):
    """Add new todos to the planning list."""
    new_todos = params.get("newTodos", [])
    _todos.extend(new_todos)
    return f"Added {len(new_todos)} todos. Total: {len(_todos)}"

def mark_todo_done(params):
    """Mark a todo as completed."""
    todo = params.get("todo")
    if todo in _todos:
        _todos.remove(todo)
        _done.append(todo)
        return f"Completed: {todo}"
    return f"Todo not found: {todo}"
```

With these planning tools, our agent can break down complex tasks, track progress, and ensure nothing gets forgotten.

## Putting It All Together: The Complete Agent

Here's how all these components work together in our main agent loop:

```python
def run_agent(goal):
    prompt = f"""
    You are a helpful assistant. Before starting any work, make a plan using your todo list.
    Use available tools to accomplish tasks. Mark todos as done when complete.
    Summarize your actions when finished.
    
    Today is {datetime.now()}
    """
    
    completion = complete_with_tools({
        "messages": [
            {"role": "developer", "content": prompt},
            {"role": "user", "content": goal},
        ],
        "model": "llama3.2:latest",
        "tool_choice": "auto",
        "tools": available_tools,
    })
    
    return completion.get("choices", [{}])[0].get("message", {}).get("content")
```

This simple function creates an agent that can:
1. Understand the goal
2. Create a plan (using todo tools)
3. Execute actions (using available tools)  
4. Track progress (marking todos complete)
5. Provide a summary

## The Magic in Action

When you run this agent with a goal like "I want to learn about building agents without a framework," here's what happens:

1. **Planning Phase**: The agent creates a todo list
   - "Research agent fundamentals"
   - "Find relevant resources" 
   - "Summarize key concepts"

2. **Execution Phase**: The agent uses tools to:
   - Search for information
   - Browse websites
   - Take notes

3. **Tracking Phase**: As each task completes, it marks todos as done

4. **Summary Phase**: Finally, it provides a comprehensive report

The beautiful part? You can watch this entire process unfold in real-time through the console output.

## What We've Learned

Building this agent from scratch has taught me several key insights:

#### 1. **Simplicity is Powerful**
You don't need complex frameworks to build effective agents. The core concepts are surprisingly straightforward.

#### 2. **The Loop is Everything** 
The while loop that continues until completion is what transforms an LLM from a chatbot into an agent.

#### 3. **Self-Evaluation is Critical**
Agents that can judge their own work are far more reliable than those that can't.

#### 4. **Tools Extend Capabilities**
The real power comes not from the LLM itself, but from the tools it can use to interact with the world.

#### 5. **Planning Makes Agents Reliable**
Agents that plan before acting are more systematic and less likely to miss important steps.

## Why This Matters for Your Career

Understanding these fundamentals makes you a better AI engineer in several ways:

- **Framework Agnostic**: You can work with any agentic framework becuase you understand the underlying principles
- **Better Debugging**: When production agents fail, you know where to look
- **Innovation Ready**: You can spot oportunities for new approaches and optimizations
- **Technical Leadership**: You can guide architectural decisions with deep understanding

## The Future of Agentic Systems

We're still in the early days of AI agents. While frameworks like LangGraph and AutoGen are fantastic, the field is evolving rapidly. New patterns, architectures, and approaches are emerging constantly.

By understanding the fundamentals - the core loop, tool integration, planning, and self-evaluation - you're positioned to adapt to whatever comes next. You're not just using AI; you're truly understanding it.

## Ready to Build Your Own?

If this post has sparked your curiosity (and I hope it has!), I encourage you to try building your own agent from scratch. You'll be amazed at how much you learn in the process.

The complete code for everything discussed in this post is available on GitHub: [https://github.com/imanoop7/agent_from_scratch](https://github.com/imanoop7/agent_from_scratch)

Start simple - build the basic LLM interface, add some conditions, integrate a tool or two, and before you know it, you'll have a working agent. Then you can expand from there.

Remember: every expert was once a beginner. The best way to truly understand AI agents is to build one yourself. So grab your code editor, fire up that local LLM, and start building!

*What will you teach your agent to do?*

---

*Have you built your own agents from scratch? I'd love to hear about your experiences and what you learned along the way. Drop me a note and let's discuss the fascinating world of agentic AI!*

{{< citation >}}