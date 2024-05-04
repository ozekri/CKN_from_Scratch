# CKN from Scratch
Implementation from scratch, without ML librairies, of the Convolutional Kernel Network of [[Mairal, 2016](https://proceedings.neurips.cc/paper_files/paper/2016/file/fc8001f834f6a5f0561080d134d53d29-Paper.pdf)].

<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://www.master-mva.com/">
    <img src="https://github.com/ozekri/CKN_from_Scratch/blob/main/img/CKN.png" alt="Logo">
  </a>
<h3 align="center">Project Kernel Methods 2023/2024 : <a href="https://github.com/ozekri/CKN_from_Scratch">CKN from Scratch</a></h3>

</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#running">running</a></li>
      </ul>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

We set ourselves the challenge of implementing **from scratch**, the **Convolutional Kernel Network** introduced in [[Mairal et al., 2014](https://proceedings.neurips.cc/paper_files/paper/2014/file/81ca0262c82e712e50c580c032d99b60-Paper.pdf)] and developed in [[Mairal, 2016](https://proceedings.neurips.cc/paper_files/paper/2016/file/fc8001f834f6a5f0561080d134d53d29-Paper.pdf)].

Since it's a from scratch implementation, it's very inefficient and doesn't work on GPUs. We advise you to use [Dexiong Chen's excellent Pytorch implementation](https://github.com/claying/CKN-Pytorch-image), which helped us build this one.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

[![Python][Python]][Python-url]


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

All you need is a Python environment with the Numpy and Scipy libraries. To visualize KMeans algorithms, you'll also need scikit-learn, but it's absolutely not necessary to run the Convolutional Kernel Network.

### Running

1. Clone the repo
   ```sh
   git clone https://github.com/ozekri/CKN_from_Scratch
   ```
2. Download the [data](https://www.kaggle.com/competitions/data-challenge-kernel-methods-2023-2024-extension/data) and put the files in the `/data` folder.
3. Run the file `start.py`. You can modify the args directly in the file, or running the following in a command prompt, for example :
   ```sh
   python start.py --epochs YOUR_NB_EPOCHS --lr YOUR_LR --alpha YOUR_REG_FACTOR --model YOUR_MODEL
   ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Don't hesitate to contribute, especially on the GPU implementation of the convolution that we didn't take the time to implement properly and integrate into our code. A few code snippets are available in the `/gpu` folder, but feel free to start from scratch!

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Oussama ZEKRI - oussama.zekri@ens-paris-saclay.fr - [Twitter](https://twitter.com/oussamazekri_)

Elyas BENYAMINA - elyas.benyamina@ens-paris-saclay.fr

[Project Link](https://github.com/ozekri/CKN_from_Scratch)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* We wanted to thank the few people who worked on the CKNs, in particular [Julien Mairal](https://lear.inrialpes.fr/people/mairal/)'s former PhD students such as [Alberto Bietti](https://alberto.bietti.me/), [Dexiong Chen](https://dexiong.me/) and [Grégoire Mialon](https://gregoiremialon.github.io/). We'd also like to mention [Zaid Harchaoui](https://scholar.google.fr/citations?user=yCyR-TsAAAAJ&hl=fr), [Vincent Roulet](https://vroulet.github.io/), [Mattis Paulin](https://lear.inrialpes.fr/people/paulin/) and others... Their work has enabled us to deepen our knowledge of CKNs.
* We want to thank our teachers [Julien Mairal](https://lear.inrialpes.fr/people/mairal/), [Michael Arbel](https://michaelarbel.github.io/) and [Alessandro Rudi](https://www.di.ens.fr/~rudi/), who enabled us to get involved in this challenge, and to learn so much about both kernel methods and ML in general.


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/elyasbny/elyasbny.svg?style=for-the-badge
[contributors-url]: https://github.com/elyasbny
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
[Python]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://www.python.org/
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
