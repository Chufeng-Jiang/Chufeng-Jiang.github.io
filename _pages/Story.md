---

permalink: /story/
title: "Story"

author_profile: true
redirect_from: 
  - /about
---
<div class="masthead">
  <div class="masthead__inner-wrap">
    <div class="masthead__menu">
      <nav id="site-nav" class="greedy-nav">
        <button><div class="navicon"></div></button>
        <ul class="visible-links">
          <li class="masthead__menu-item masthead__menu-item--lg masthead__menu-home-item"><a href="{{ site.url }}">Homepage</a></li>
          {% for link in site.data.navigation.main %}
            <li class="masthead__menu-item"><a href="{{ site.url }}{{ link.url }}">{{ link.title }}</a></li>
			<li class="masthead__menu-item"><a href="{{ domain }}{{ link.url }}"></a></li>
          {% endfor %}
		</ul>
        <ul class="hidden-links hidden"></ul>
      </nav>
    </div>
  </div>
</div>

{% if site.google_scholar_stats_use_cdn %}
{% assign gsDataBaseUrl = "https://cdn.jsdelivr.net/gh/" | append: site.repository | append: "@" %}
{% else %}
{% assign gsDataBaseUrl = "https://raw.githubusercontent.com/" | append: site.repository | append: "/" %}
{% endif %}
{% assign url = gsDataBaseUrl | append: "google-scholar-stats/gs_data_shieldsio.json" %}

<span class='anchor' id='about-me'></span>



<!--If you like the template of this homepage, welcome to star and fork my open-sourced template version [AcadHomepage ![](https://img.shields.io/github/stars/RayeRen/acad-homepage.github.io?style=social)](https://github.com/RayeRen/acad-homepage.github.io).-->
<!-- excerpt: ""-->

{% include_relative story/cs.md %}
