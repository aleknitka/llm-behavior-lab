# Purchase Decision-Making Inventory (PDMI)

This module codes the Purchase Decision-Making Inventory from:

Soler Anguiano, F. L., Bustos Aguayo, J. M., Palacios, J., Zeelenberg, M., &
Diaz Loving, R. (2019). Development and validation of the Inventory of Emotional
and Reasoned Purchases Decision-Making Styles (PDMI). Suma Psicologica, 26(2),
75-85.

- DOI: `10.14349/sumapsi.2019.v26.n2.3`
- Source URL: <http://dx.doi.org/10.14349/sumapsi.2019.v26.n2.3>
- Licence: CC BY-NC-ND 4.0 as stated by the source article.

## Response Format

PDMI uses a five-point frequency response scale:

1. Never
2. Rarely
3. Sometimes
4. Often
5. Always

## Sections

The coded questionnaire contains 50 items in the source order from the EFA tables:

- `pdmi_emotional_items`: 30 emotional purchase decision-making items from Table 2.
- `pdmi_reasoned_items`: 20 reasoned purchase decision-making items from Table 3.

## Subscales

The emotional section contains five subscales:

- `impulsivity`
- `indebtedness`
- `negative_emotions`
- `frustration`
- `hedonism`

The reasoned section contains three subscales:

- `saving`
- `reasoning`
- `search_of_information`

Factor loadings are stored on each item and item mapping. Cronbach alpha and
variance explained values are stored on each `Scale.metadata` entry.

Executable scoring models are intentionally not included yet. The questionnaire
definition preserves source subscale mappings for future scoring work, while
runtime scoring output remains deferred.

## Cultural and Sample Context

The PDMI was developed for consumer decision-making styles in Mexico. The source
study used a total sample of 518 Mexican participants, split into 300 participants
for exploratory factor analysis and 218 for confirmatory factor analysis. The
authors describe the instrument as culturally relevant for the Mexican context and
recommend future work with other Mexican and Latin American populations.
