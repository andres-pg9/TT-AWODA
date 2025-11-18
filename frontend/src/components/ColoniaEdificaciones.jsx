// ColoniaEdificaciones.jsx
import React from "react";

const ColoniaEdificaciones = ({ edificaciones }) => {
  if (!edificaciones || edificaciones.length === 0) {
    return null;
  }

  return (
    <>
      {edificaciones.map((ed, index) => (
        <div key={index} className="awoda-distribucion-item">
          <span className="awoda-distribucion-numero">
            {ed.ranking ?? index + 1}.
          </span>
          <span className="awoda-distribucion-nombre">
            {ed.nombre || ed.tipo}
          </span>
        </div>
      ))}
    </>
  );
};

export default ColoniaEdificaciones;
