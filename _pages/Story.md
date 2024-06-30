---
layout: default
permalink: /story/
title: "Story"

author_profile: true
redirect_from: 
  - /about
---
  <head>
    {% include head.html %}
    {% include head/custom.html %}
  </head>

  <body>
  
{% if site.google_scholar_stats_use_cdn %}
{% assign gsDataBaseUrl = "https://cdn.jsdelivr.net/gh/" | append: site.repository | append: "@" %}
{% else %}
{% assign gsDataBaseUrl = "https://raw.githubusercontent.com/" | append: site.repository | append: "/" %}
{% endif %}
{% assign url = gsDataBaseUrl | append: "google-scholar-stats/gs_data_shieldsio.json" %}

<span class='anchor' id='about-me'></span>



If you like the template of this homepage, welcome to star and fork my open-sourced template version [AcadHomepage ![](https://img.shields.io/github/stars/RayeRen/acad-homepage.github.io?style=social)](https://github.com/RayeRen/acad-homepage.github.io).
<!-- excerpt: ""-->

{% include_relative story/cs.md %}


    {% include browser-upgrade.html %}
    {% include masthead.html %}

    <div id="main" role="main">
      {% include sidebar.html %}
    
      <article class="page" itemscope itemtype="http://schema.org/CreativeWork">
        {% if page.title %}<meta itemprop="headline" content="{{ page.title | markdownify | strip_html | strip_newlines | escape_once }}">{% endif %}
        <div class="page__inner-wrap">
          <section class="page__content" itemprop="text">
If you like the template of this homepage, welcome to star and fork my open-sourced template version [AcadHomepage ![](https://img.shields.io/github/stars/RayeRen/acad-homepage.github.io?style=social)](https://github.com/RayeRen/acad-homepage.github.io).
<!-- excerpt: ""-->

{% include_relative story/cs.md %}
          </section>
        </div>
      </article>
    </div>

    {% include scripts.html %}

  </body>
