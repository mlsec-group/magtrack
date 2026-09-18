## Ethical Considerations

We are aware that research into attacks carries the potential for misuse.
However, this risk must be weighed against the necessity of understanding emerging threats in order to evaluate their real-world impact and improve the security of widely deployed systems.
Following the principles of the Menlo Report, in particular beneficence, respect for persons, and justice, we designed and conducted our study to minimize harm while maximizing societal benefit.
Our research institute does not operate an Institutional Review Board (IRB), as is common in Europe.
Nevertheless, we followed established best practices to avoid harm to stakeholders and to protect personal data throughout the study.

#### Stakeholders

Our research affects multiple stakeholders with varying degrees of exposure and risk.
First, society at large and smartphone users are indirectly impacted, as our work demonstrates that commonly accessible magnetometer data can be used to infer sensitive information such as transport mode and colocation.
Second, participants involved in our dataset collection are directly affected, as they contributed sensor data that inherently reflects their travel behavior and movement patterns.
Third, the research community and industry stakeholders (e.g., smartphone manufacturers and OS developers) are impacted by the publication of our findings, as they can use these insights to design improved privacy protections and sensor access controls.
Finally, potential adversaries are also stakeholders, as they could misuse the presented techniques to infer user behavior without consent.

#### Privacy of participants

To respect the rights and privacy of participants, we ensured that participation was voluntary and based on informed consent.
All data was processed and analyzed in a controlled environment.
In the published dataset, we further reduced re-identification risks by pseudonymizing stations and segments and by transforming timestamps into relative values.
We note that, despite these precautions, we cannot guarantee complete protection against deanonymization.
We explicitly informed participants about this residual risk during the consent process.
These measures mitigate harms, such as privacy loss of participants.

#### Results and Impact

Our work reveals a security and privacy threat posed by magnetometer data from smartphones.
A potential harm of publication is that adversaries may replicate or extend our techniques.
At the same time, the benefits include improved understanding of sensor-based privacy risks and the development of countermeasures, such as restricting sensor access or reducing sampling rates.
We therefore argue that the expected societal benefits outweigh the potential risks.

#### Decision

We decided to conduct and publish this research based on a careful evaluation of risks and benefits.
From a beneficence perspective, the work contributes to identifying and mitigating a realistic privacy threat affecting a large population of smartphone users.
From a respect-for-persons perspective, we ensured that participant rights were preserved through informed consent and privacy-preserving data handling.
While the possibility of misuse cannot be fully eliminated, we believe that disclosure of the threat is necessary to enable effective safeguards.
Overall, we conclude that both conducting and publishing this research is ethically justified.