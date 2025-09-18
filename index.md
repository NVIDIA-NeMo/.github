# NVIDIA NeMo Framework

NVIDIA NeMo Framework is an end-to-end training framework for large language models (LLMs), multi-modal models and speech models designed to run on NVIDIA accelerated infrastructure. It enables seamless scaling of both pre-training and post-training workloads from single GPU to thousand-node clusters for Hugging Face, Megatron, and PyTorch models.

This site hosts developer updates, tutorials, and insights about NeMo's latest core components and innovations.

## Latest Blog Posts

{% set latest_posts = get_latest_blog_posts(3) %}
{% if latest_posts %}
{% for post in latest_posts %}
### [{{ post.title }}]({{ post.url }})

*{{ post.date.strftime('%B %d, %Y') }}*

{% endfor %}
{% endif %}

---

[View all blog posts →](blog/index.md){ .md-button }

## NeMo Framework Components

<div class="grid cards" markdown>

-   🚀 __NeMo-RL__

    ---

    Scalable toolkit for efficient model reinforcement learning and post-training. Includes algorithms like DPO, GRPO, and support for everything from single-GPU prototypes to thousand-GPU deployments.

    [🚀 GitHub Repository](https://github.com/NVIDIA-NeMo/RL){ .md-button }

    [📖 Documentation](https://docs.nvidia.com/nemo/rl/latest/index.html){ .md-button }

-   🚀 __NeMo-Automodel__

    ---

    Day-0 support for any Hugging Face model leveraging PyTorch native functionalities while providing performance and memory optimized training and inference recipes.

    [🚀 GitHub Repository](https://github.com/NVIDIA-NeMo/Automodel){ .md-button }

    [📖 Documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/automodel/index.html){ .md-button }

</div>

## License

Apache 2.0 licensed with third-party attributions documented in each repository.
