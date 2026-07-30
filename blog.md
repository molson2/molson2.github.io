---
layout: page
title: Blog
permalink: /blog/
---

<div class="post-list">
  {% for post in site.posts %}
    <article style="margin-bottom: 2.25rem;">
      <h2 style="margin-bottom: 0.25rem;">
        <a href="{{ post.url | relative_url }}" style="text-decoration: none;">
          {{ post.title }}
        </a>
      </h2>

      <p style="color: #666; margin-top: 0;">
        {{ post.date | date: "%B %-d, %Y" }}
      </p>

      <p>
        <a href="{{ post.url | relative_url }}">Read more →</a>
      </p>
    </article>
  {% endfor %}
</div>
