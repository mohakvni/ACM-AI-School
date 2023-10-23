import React from 'react'
import style from './slides.module.css'

export const Slides : React.FC<{link: string}> = ({link}) => {
  return (
    <div className={style.slides}>
        <iframe src={link}  width="600" height="377" allowFullScreen={true}/>
    </div>
  )
}

export default Slides
